import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import psutil

import config

STATE_LOCK = threading.Lock()


def _console_text(value):
    value = str(value).replace("ñ", "n").replace("Ñ", "N")
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


def _display_path(value):
    path = Path(value)
    try:
        return path.resolve().relative_to(config.PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _now():
    buenos_aires = timezone(timedelta(hours=-3), name="UTC-03:00")
    return datetime.now(buenos_aires).isoformat(timespec="seconds")


def _clear_schedule_artifacts():
    log_dir = config.PIPELINE_LOGS
    if config.PIPELINE_LOGS.exists():
        try:
            shutil.rmtree(config.PIPELINE_LOGS)
        except OSError:
            run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S") + f"_{os.getpid()}"
            log_dir = config.PIPELINE_LOGS / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    for path in (config.PIPELINE_STATE, config.PIPELINE_STATE.with_suffix(".tmp"), config.STOP_FILE):
        path.unlink(missing_ok=True)
    return log_dir


def _read_state():
    if not config.PIPELINE_STATE.exists():
        return {}
    try:
        return json.loads(config.PIPELINE_STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state(state):
    config.OPTIMIZATION_RESULTS.mkdir(parents=True, exist_ok=True)
    temporary_path = config.PIPELINE_STATE.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(config.PIPELINE_STATE)


def _pid_is_running(pid):
    if not pid:
        return False
    try:
        return psutil.pid_exists(int(pid))
    except (psutil.Error, ValueError):
        return False


def _terminate_process_tree(process, timeout=20):
    try:
        root = psutil.Process(process.pid)
        descendants = root.children(recursive=True)
    except (psutil.Error, OSError):
        descendants = []
        root = None

    for child in reversed(descendants):
        try:
            child.terminate()
        except psutil.Error:
            pass
    if root is not None:
        try:
            root.terminate()
        except psutil.Error:
            pass

    alive = []
    if root is not None:
        _, alive = psutil.wait_procs([*descendants, root], timeout=timeout)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass
    if alive:
        psutil.wait_procs(alive, timeout=5)
    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        process.kill()
        process.wait()


def _file_is_valid(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("rb") as file:
            prefix = file.read(128)
        return not prefix.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _poetry_command(command, *args):
    poetry = shutil.which("poetry") or "poetry"
    return [poetry, "run", command, *args]


def _stages(sgd_jobs, full_evaluation, include_contextual, overwrite=False):
    stages = [
        {
            "id": "extract",
            "command": _poetry_command("extractor", *( ["--overwrite"] if overwrite else [] )),
            "outputs": [config.dataset_raw],
            "batch": "setup",
        },
        {
            "id": "explore",
            "command": _poetry_command("explore"),
            "outputs": [],
            "batch": "setup",
        },
        {
            "id": "clean",
            "command": _poetry_command("clean"),
            "outputs": [config.dataset_clean],
            "batch": "setup",
        },
        {
            "id": "nltk_resources",
            "command": None,
            "outputs": [],
            "batch": "setup",
        },
        {
            "id": "bow",
            "command": _poetry_command("bow"),
            "outputs": [config.bow_vect, config.bow_vocab],
            "batch": "sparse_embeddings",
        },
        {
            "id": "idf_monogram",
            "command": _poetry_command("idfMonogram"),
            "outputs": [config.idf_monogram, config.DATA / "idf_monogram_vocab.csv"],
            "batch": "sparse_embeddings",
        },
        {
            "id": "idf_bigram",
            "command": _poetry_command("idfBigram"),
            "outputs": [config.idf_bigram, config.DATA / "idf_bigram_vocab.csv"],
            "batch": "sparse_embeddings",
        },
        {
            "id": "fasttext",
            "command": _poetry_command("fasttext"),
            "outputs": [config.fasttext_embeddings, config.fasttext_vocab],
            "batch": "sparse_embeddings",
        },
    ]
    if include_contextual:
        stages.extend([
            {
                "id": "bert",
                "command": _poetry_command("bert"),
                "outputs": [config.bert_vect, config.vocabulario_bert],
                "batch": "contextual_embeddings",
            },
            {
                "id": "elmo",
                "command": _poetry_command("elmo", "--download-model"),
                "outputs": [config.elmo_vect, config.vocabulario_elmo],
                "batch": "contextual_embeddings",
            },
        ])
    stages.append({
        "id": "sgd",
        "command": _poetry_command(
            "sgd",
            "--jobs", str(sgd_jobs),
            "--no-plots",
            *( ["--full-evaluation"] if full_evaluation else [] ),
        ),
        "outputs": [
            config.df_sb_sgd,
            config.df_b_sgd,
            config.df_optimizado_sb_sgd,
            config.df_optimizado_b_sgd,
        ],
        "batch": "training",
    })
    return stages


def _run_stage(stage, state, log_dir, overwrite=False):
    stage_id = stage["id"]
    existing_outputs = [path for path in stage["outputs"] if Path(path).exists()]
    if not overwrite and existing_outputs:
        if all(_file_is_valid(path) for path in stage["outputs"]):
            print(f"[{stage_id}] salidas existentes verificadas; se omite.", flush=True)
            with STATE_LOCK:
                state["stages"][stage_id] = {
                    "status": "done", "finished_at": _now(), "skipped_existing": True,
                }
                _write_state(state)
            return True

        message = (
            f"[{stage_id}] hay salidas existentes o incompletas; no se modifican. "
            "Vuelve a ejecutar con --overwrite para regenerarlas."
        )
        print(_console_text(message), file=sys.stderr, flush=True)
        with STATE_LOCK:
            state["stages"][stage_id] = {
                "status": "blocked", "finished_at": _now(),
                "existing_outputs": [str(path) for path in existing_outputs],
                "error": message,
            }
            _write_state(state)
        return False

    if config.STOP_FILE.exists():
        with STATE_LOCK:
            state["stages"][stage_id] = {
                "status": "stopped", "finished_at": _now(),
            }
            _write_state(state)
        return False

    log_path = log_dir / f"{stage_id}.log"
    with STATE_LOCK:
        state["current_stage"] = stage_id
        state["stages"][stage_id] = {
            "status": "running", "started_at": _now(), "log": str(log_path),
        }
        _write_state(state)
    print(f"[{stage_id}] iniciando; log: {_display_path(log_path)}", flush=True)

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    if stage["command"] is None:
        try:
            with log_path.open("a", encoding="utf-8") as log_file:
                with redirect_stdout(log_file), redirect_stderr(log_file):
                    from utils import verificar_recursos_nltk
                    verificar_recursos_nltk()
            return_code = 0
        except Exception as error:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{type(error).__name__}: {error}\n")
            return_code = 1
    else:
        with log_path.open("a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                stage["command"],
                cwd=config.PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                env=child_env,
            )
            while process.poll() is None:
                if config.STOP_FILE.exists():
                    _terminate_process_tree(process)
                    with STATE_LOCK:
                        state["stages"][stage_id].update({"status": "stopped", "finished_at": _now()})
                        _write_state(state)
                    return False
                time.sleep(1)
        return_code = process.returncode

    missing_outputs = [
        _display_path(path) for path in stage["outputs"] if not _file_is_valid(path)
    ]
    if return_code == 0 and missing_outputs:
        return_code = 2
    with STATE_LOCK:
        state["stages"][stage_id].update({
            "status": "done" if return_code == 0 else "failed",
            "finished_at": _now(),
            "return_code": return_code,
            "missing_outputs": missing_outputs,
        })
        _write_state(state)
    if return_code != 0:
        print(
            f"[{stage_id}] fallo con codigo {return_code}; revisa {_display_path(log_path)}",
            flush=True,
        )
        return False
    print(f"[{stage_id}] terminado.", flush=True)
    return True


def _run_pipeline(*, workers=1, sgd_jobs=1, full_evaluation=False, include_contextual=True, notify=True, overwrite=False, log_dir=None):
    log_dir = Path(log_dir) if log_dir else config.PIPELINE_LOGS
    log_dir.mkdir(parents=True, exist_ok=True)
    config.STOP_FILE.unlink(missing_ok=True)
    state = _read_state()
    stages = _stages(sgd_jobs, full_evaluation, include_contextual, overwrite)
    state.update({
        "status": "running",
        "pid": os.getpid(),
        "started_at": _now(),
        "workers": workers,
        "sgd_jobs": sgd_jobs,
        "full_evaluation": full_evaluation,
        "include_contextual": include_contextual,
        "notify": notify,
        "overwrite": overwrite,
        "stages": {stage["id"]: {"status": "pending"} for stage in stages},
    })
    _write_state(state)
    batches = []
    for stage in stages:
        if not batches or batches[-1][0]["batch"] != stage["batch"]:
            batches.append([stage])
        else:
            batches[-1].append(stage)

    success = True
    try:
        for batch in batches:
            if len(batch) == 1 or workers == 1:
                for stage in batch:
                    if not _run_stage(stage, state, log_dir, overwrite):
                        success = False
                        break
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    outcomes = list(pool.map(
                        lambda stage: _run_stage(stage, state, log_dir, overwrite),
                        batch,
                    ))
                success = all(outcomes)
            if not success:
                break
    except BaseException as error:
        state["error"] = f"{type(error).__name__}: {error}"
        success = False
        raise
    finally:
        state["status"] = "completed" if success else (
            "stopped" if config.STOP_FILE.exists() else "failed"
        )
        state["finished_at"] = _now()
        state.pop("current_stage", None)
        _write_state(state)
        config.STOP_FILE.unlink(missing_ok=True)
        _notify_windows(state)
    return 0 if success else 1


def _notify_windows(state):
    if os.name != "nt" or not state.get("notify", True):
        return
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        return
    succeeded = state.get("status") == "completed"
    title = "Pipeline completado" if succeeded else "Pipeline detenido o fallido"
    message = "Todas las etapas finalizaron." if succeeded else "Revisa el estado y los logs de schedule."
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "Add-Type -AssemblyName System.Drawing; "
        "$n = New-Object System.Windows.Forms.NotifyIcon; "
        "$n.Icon = [System.Drawing.SystemIcons]::Information; "
        f"$n.BalloonTipTitle = '{title}'; "
        f"$n.BalloonTipText = '{message}'; "
        "$n.Visible = $true; "
        "$n.ShowBalloonTip(8000); "
        "Start-Sleep -Milliseconds 8500; $n.Dispose()"
    )
    try:
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        subprocess.Popen(
            [powershell, "-NoProfile", "-STA", "-WindowStyle", "Hidden", "-Command", script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
    except OSError:
        pass


def _start_background(args):
    state = _read_state()
    if state.get("status") in {"running", "starting"} and _pid_is_running(state.get("pid")):
        print(f"Ya hay un pipeline activo (PID {state['pid']}).")
        return 1

    log_dir = _clear_schedule_artifacts()
    log_path = log_dir / "scheduler.log"
    command = _poetry_command(
        "schedule", "_worker",
        "--log-dir", str(log_dir),
        "--workers", str(args.workers),
        "--sgd-jobs", str(args.sgd_jobs),
    )
    if args.full_evaluation:
        command.append("--full-evaluation")
    if args.skip_contextual:
        command.append("--skip-contextual")
    if args.no_notify:
        command.append("--no-notify")
    if args.overwrite:
        command.append("--overwrite")

    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        )
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=config.PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            env=child_env,
        )
    state = {
        "status": "starting",
        "pid": process.pid,
        "log": str(log_path),
        "started_at": _now(),
    }
    _write_state(state)
    print(f"Pipeline iniciado en segundo plano (PID {process.pid}).")
    print(f"Estado: poetry run schedule status")
    print(f"Log: {_display_path(log_path)}")
    return 0


def _status():
    state = _read_state()
    if not state:
        print("No hay ejecuciones registradas.")
        return 1
    status = state.get("status", "unknown")
    missing_by_stage = {}
    if status == "running" and not _pid_is_running(state.get("pid")):
        status = "worker_no_activo; estado guardado: running"
    elif status == "completed":
        for stage in _stages(
            state.get("sgd_jobs", 1),
            state.get("full_evaluation", False),
            state.get("include_contextual", True),
        ):
            recorded = state.get("stages", {}).get(stage["id"], {})
            if recorded.get("status") == "done":
                missing = [
                    _display_path(path)
                    for path in stage["outputs"]
                    if not _file_is_valid(path)
                ]
                if missing:
                    missing_by_stage[stage["id"]] = missing
        if missing_by_stage:
            status = "completed_inconsistent; outputs missing"
    print(f"Estado: {status}")
    print(f"PID: {state.get('pid', '-')}")
    print(f"Etapa: {state.get('current_stage', '-')}")
    print(f"Inicio: {state.get('started_at', '-')}")
    print(f"Log: {_display_path(state.get('log', config.PIPELINE_LOGS / 'scheduler.log'))}")
    for stage_id, stage in state.get("stages", {}).items():
        print(f"  {stage_id}: {stage.get('status')}")
        for missing in missing_by_stage.get(stage_id, []):
            print(f"    salida faltante: {missing}")
    return 1 if status in {"failed", "stopped"} or status.startswith(("worker_no_activo", "completed_inconsistent")) else 0


def _stop():
    state = _read_state()
    if state.get("status") != "running" or not _pid_is_running(state.get("pid")):
        print("No hay un pipeline activo para detener.")
        return 1
    config.STOP_FILE.write_text("stop\n", encoding="utf-8")
    print("Se solicito detener el pipeline. La etapa actual terminara de forma controlada.")
    return 0


def _gui():
    try:
        import tkinter as tk
        from tkinter import messagebox, scrolledtext, ttk
    except ImportError as error:
        print(f"No se pudo cargar Tkinter: {_console_text(error)}", file=sys.stderr)
        return 1

    root = tk.Tk()
    root.title("Pipeline — estado de ejecución")
    root.geometry("920x680")
    root.minsize(720, 520)

    status_text = tk.StringVar(value="Consultando estado…")
    current_stage_text = tk.StringVar(value="Etapa activa: —")
    include_contextual = tk.BooleanVar(value=True)
    overwrite = tk.BooleanVar(value=False)

    header = ttk.Frame(root, padding=12)
    header.pack(fill="x")
    ttk.Label(header, textvariable=status_text, font=("Segoe UI", 12, "bold")).pack(anchor="w")
    ttk.Label(header, textvariable=current_stage_text).pack(anchor="w", pady=(4, 0))

    table = ttk.Treeview(root, columns=("stage", "status", "started", "finished"), show="headings", height=10)
    table.heading("stage", text="Etapa")
    table.heading("status", text="Estado")
    table.heading("started", text="Inicio")
    table.heading("finished", text="Fin")
    table.column("stage", width=150)
    table.column("status", width=110, anchor="center")
    table.column("started", width=180)
    table.column("finished", width=180)
    table.pack(fill="both", expand=True, padx=12, pady=4)

    log_frame = ttk.LabelFrame(root, text="Log de la etapa activa", padding=8)
    log_frame.pack(fill="both", expand=True, padx=12, pady=(4, 0))
    log_view = scrolledtext.ScrolledText(log_frame, height=12, wrap="word", state="disabled")
    log_view.pack(fill="both", expand=True)
    active_log_path = None
    log_offset = 0

    controls = ttk.Frame(root, padding=12)
    controls.pack(fill="x")
    ttk.Checkbutton(controls, text="Incluir BERT y ELMo", variable=include_contextual).pack(side="left")
    ttk.Checkbutton(controls, text="Sobrescribir salidas", variable=overwrite).pack(side="left", padx=(10, 0))

    def start_pipeline():
        nonlocal active_log_path, log_offset
        options = argparse.Namespace(
            workers=1,
            sgd_jobs=1,
            full_evaluation=False,
            skip_contextual=not include_contextual.get(),
            no_notify=False,
            overwrite=overwrite.get(),
        )
        active_log_path = None
        log_offset = 0
        log_view.configure(state="normal")
        log_view.delete("1.0", "end")
        log_view.configure(state="disabled")
        try:
            _start_background(options)
        except (OSError, RuntimeError) as error:
            messagebox.showerror("No se pudo iniciar el schedule", str(error), parent=root)

    ttk.Button(controls, text="Iniciar", command=start_pipeline).pack(side="right", padx=(8, 0))
    ttk.Button(controls, text="Detener", command=_stop).pack(side="right")

    def refresh():
        nonlocal active_log_path, log_offset
        state = _read_state()
        current_status = state.get("status", "sin ejecuciones")
        if current_status == "completed":
            for stage in _stages(
                state.get("sgd_jobs", 1),
                state.get("full_evaluation", False),
                state.get("include_contextual", True),
            ):
                recorded = state.get("stages", {}).get(stage["id"], {})
                if recorded.get("status") == "done" and not all(_file_is_valid(path) for path in stage["outputs"]):
                    current_status = "completed_inconsistent; faltan salidas"
                    break
        status_text.set(f"Estado: {current_status} | PID: {state.get('pid', '—')}")
        current_stage_text.set(f"Etapa activa: {state.get('current_stage', '—')}")
        stage_records = state.get("stages", {})
        stage_order = [stage["id"] for stage in _stages(
            state.get("sgd_jobs", 1),
            state.get("full_evaluation", False),
            state.get("include_contextual", True),
            state.get("overwrite", False),
        )]
        stage_order.extend(stage_id for stage_id in stage_records if stage_id not in stage_order)
        current_ids = set()
        for index, stage_id in enumerate(stage_order):
            if stage_id not in stage_records:
                continue
            stage = stage_records[stage_id]
            current_ids.add(stage_id)
            values = (
                stage_id,
                stage.get("status", "—"),
                stage.get("started_at", "—"),
                stage.get("finished_at", "—"),
            )
            if table.exists(stage_id):
                table.item(stage_id, values=values)
            else:
                table.insert("", "end", iid=stage_id, text=stage_id, values=values)
            table.move(stage_id, "", index)
        for item_id in table.get_children():
            if item_id not in current_ids:
                table.delete(item_id)

        active_stage = state.get("current_stage")
        stage_log = stage_records.get(active_stage, {}).get("log") if active_stage else None
        if stage_log != active_log_path:
            active_log_path = stage_log
            log_offset = 0
            log_view.configure(state="normal")
            log_view.delete("1.0", "end")
            log_view.configure(state="disabled")
        if active_log_path:
            try:
                if Path(active_log_path).stat().st_size < log_offset:
                    log_offset = 0
                    log_view.configure(state="normal")
                    log_view.delete("1.0", "end")
                    log_view.configure(state="disabled")
                with Path(active_log_path).open("rb") as log_file:
                    log_file.seek(log_offset)
                    new_content = log_file.read(128 * 1024)
                    log_offset = log_file.tell()
                if new_content:
                    log_view.configure(state="normal")
                    log_view.insert("end", new_content.decode("utf-8", errors="replace").replace("\r", "\n"))
                    line_count = int(log_view.index("end-1c").split(".")[0])
                    if line_count > 5000:
                        log_view.delete("1.0", f"{line_count - 3000}.0")
                    log_view.see("end")
                    log_view.configure(state="disabled")
            except OSError:
                pass
        root.after(1500, refresh)

    refresh()
    root.mainloop()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Programa y supervisa el pipeline de procesamiento.")
    commands = parser.add_subparsers(dest="action", required=True)

    for action in ("start", "run"):
        command = commands.add_parser(action, help="Inicia la ejecucion completa" if action == "start" else "Ejecuta en primer plano")
        command.add_argument("--workers", type=int, default=1, help="Vectorizadores concurrentes (default: 1).")
        command.add_argument("--sgd-jobs", type=int, default=1, help="Procesos usados por curvas SGD (default: 1).")
        command.add_argument("--full-evaluation", action="store_true", help="SGD usa 5 folds y 5 puntos de curva; mas lento.")
        command.add_argument("--skip-contextual", action="store_true", help="Omite BERT y ELMo, que no son requeridos por SGD.")
        command.add_argument("--no-notify", action="store_true", help="Desactiva el aviso del area de notificacion al terminar.")
        command.add_argument(
            "--overwrite", action="store_true",
            help="Regenera las salidas existentes; por defecto se conservan y se omiten.",
        )

    commands.add_parser("status", help="Muestra estado y etapas")
    commands.add_parser("stop", help="Solicita una detencion controlada")
    commands.add_parser("plan", help="Lista las etapas incluidas en el flujo completo")
    commands.add_parser("gui", help="Abre el panel grafico de estado y control")
    worker = commands.add_parser("_worker")
    worker.add_argument("--workers", type=int, default=1)
    worker.add_argument("--sgd-jobs", type=int, default=1)
    worker.add_argument("--full-evaluation", action="store_true")
    worker.add_argument("--skip-contextual", action="store_true")
    worker.add_argument("--no-notify", action="store_true")
    worker.add_argument("--overwrite", action="store_true")
    worker.add_argument("--log-dir", type=Path)
    args = parser.parse_args(argv)

    if args.action in {"start", "run", "_worker"}:
        if args.workers < 1 or args.sgd_jobs < 1:
            parser.error("workers y sgd-jobs deben ser enteros positivos")
        if args.action == "start":
            return _start_background(args)
        if args.action == "run":
            log_dir = _clear_schedule_artifacts()
        else:
            log_dir = args.log_dir
        return _run_pipeline(
            workers=args.workers,
            sgd_jobs=args.sgd_jobs,
            full_evaluation=args.full_evaluation,
            include_contextual=not args.skip_contextual,
            notify=not args.no_notify,
            overwrite=args.overwrite,
            log_dir=log_dir,
        )
    if args.action == "status":
        return _status()
    if args.action == "plan":
        for number, stage in enumerate(_stages(1, False, True), start=1):
            print(f"{number}. {stage['id']}")
        return 0
    if args.action == "gui":
        return _gui()
    return _stop()


if __name__ == "__main__":
    raise SystemExit(main())
