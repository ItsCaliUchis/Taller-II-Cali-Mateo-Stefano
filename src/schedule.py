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


def _load_catalog():
    try:
        catalog = json.loads(config.JSON_COMMANDS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"No se pudo leer el catálogo de comandos: {error}") from error
    return catalog.get("commands", {})


def _default_plan():
    commands = []
    for name, info in _load_catalog().items():
        metadata = info.get("schedule")
        if info.get("status") == "active" and isinstance(metadata, dict):
            commands.append({"name": name, "args": list(metadata.get("default_args", []))})
    return {"version": 1, "commands": commands}


def _save_plan(plan, path=None):
    path = Path(path or config.PIPELINE_PLAN)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _load_plan(path=None):
    path = Path(path or config.PIPELINE_PLAN)
    if not path.exists():
        plan = _default_plan()
        _save_plan(plan, path)
        return plan
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"No se pudo leer el plan {path}: {error}") from error
    if not isinstance(plan, dict) or plan.get("version") != 1 or not isinstance(plan.get("commands"), list):
        raise ValueError("Formato inválido: el plan debe tener version 1 y una lista commands.")
    catalog = _load_catalog()
    seen = set()
    for item in plan["commands"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("Cada entrada del plan debe tener un campo name.")
        name = item["name"]
        if name in seen:
            raise ValueError(f"El comando '{name}' está repetido en el plan.")
        seen.add(name)
        info = catalog.get(name)
        if not info or info.get("status") != "active" or not isinstance(info.get("schedule"), dict):
            raise ValueError(f"'{name}' no existe o no está habilitado para schedule en commands.info.json.")
        if not isinstance(item.get("args", []), list) or not all(isinstance(arg, str) for arg in item.get("args", [])):
            raise ValueError(f"Los argumentos de '{name}' deben ser una lista de textos.")
    return plan


def _stages(plan, sgd_jobs=1, full_evaluation=False, include_contextual=True, overwrite=False):
    catalog = _load_catalog()
    stages = []
    selected = []
    for item in plan["commands"]:
        name = item["name"]
        if not include_contextual and name in {"bert", "elmo"}:
            continue
        selected.append(name)
    positions = {name: index for index, name in enumerate(selected)}
    def argument_value(args, option):
        for index, value in enumerate(args):
            if value.startswith(option + "="):
                return value.split("=", 1)[1]
            if value == option and index + 1 < len(args):
                return args[index + 1]
        return None

    def selected_values(mapping, args, defaults=()):
        result = list(defaults)
        for option, choices in mapping.items():
            value = argument_value(args, option)
            if value in choices:
                result.extend(choices[value])
        return list(dict.fromkeys(result))

    args_by_name = {
        item["name"]: list(item.get("args", catalog[item["name"]]["schedule"].get("default_args", [])))
        for item in plan["commands"]
    }
    resolved_dependencies = {}
    inherited_limits = {}
    for name in selected:
        metadata = catalog[name]["schedule"]
        resolved_dependencies[name] = selected_values(
            metadata.get("depends_on_by_argument", {}), args_by_name[name], metadata.get("depends_on", [])
        )
        consumer_limit = argument_value(args_by_name[name], "--limit")
        if consumer_limit is not None:
            for dependency in resolved_dependencies[name]:
                inherited_limits.setdefault(dependency, consumer_limit)

    for name in selected:
        info = catalog[name]
        metadata = info["schedule"]
        for dependency in resolved_dependencies[name]:
            if dependency in positions:
                if positions[dependency] < positions[name]:
                    continue
                raise ValueError(f"'{dependency}' debe aparecer antes que '{name}' en plan.json.")
            dependency_info = catalog.get(dependency, {})
            dependency_schedule = dependency_info.get("schedule") or {}
            dependency_args = args_by_name.get(dependency, [])
            dependency_limit = argument_value(dependency_args, "--limit") or inherited_limits.get(dependency)
            dependency_output_names = selected_values(
                dependency_schedule.get("outputs_by_argument", {}), dependency_args,
                dependency_schedule.get("outputs", []),
            )
            dependency_outputs = [config.PROJECT_ROOT / path for path in dependency_output_names]
            if dependency_limit is not None:
                dependency_outputs = [config.ruta_limitada(path, int(dependency_limit)) for path in dependency_outputs]
            if not dependency_outputs or not all(_file_is_valid(path) for path in dependency_outputs):
                raise ValueError(
                    f"'{name}' requiere '{dependency}' antes en plan.json, o que sus salidas ya existan y sean válidas."
                )

    item_by_name = {item["name"]: item for item in plan["commands"]}
    for name in selected:
        info = catalog[name]
        metadata = info["schedule"]
        args = list(item_by_name[name].get("args", metadata.get("default_args", [])))
        if "--limit" not in args and name in inherited_limits:
            args.extend(["--limit", inherited_limits[name]])
        dependencies = resolved_dependencies[name]
        output_names = selected_values(metadata.get("outputs_by_argument", {}), args, metadata.get("outputs", []))
        limit_value = argument_value(args, "--limit")
        try:
            limit = int(limit_value) if limit_value is not None else None
        except ValueError as error:
            raise ValueError(f"--limit de '{name}' debe ser un entero positivo.") from error
        if limit is not None and limit <= 0:
            raise ValueError(f"--limit de '{name}' debe ser un entero positivo.")
        output_paths = [config.PROJECT_ROOT / path for path in output_names]
        if limit is not None:
            output_paths = [config.ruta_limitada(path, limit) for path in output_paths]
        overwrite_arg = metadata.get("overwrite_arg")
        if overwrite_arg and not overwrite:
            args = [arg for arg in args if arg != overwrite_arg]
        if name == "sgd":
            if "--jobs" not in args:
                args.extend(["--jobs", str(sgd_jobs)])
            if "--no-plots" not in args:
                args.append("--no-plots")
            if full_evaluation and "--full-evaluation" not in args:
                args.append("--full-evaluation")
        if overwrite and overwrite_arg and overwrite_arg not in args:
            args.append(overwrite_arg)
        stages.append({
            "id": name,
            "command": _poetry_command(name, *args),
            "outputs": output_paths,
            "depends_on": dependencies,
            "allow_partial_outputs": bool(metadata.get("allow_partial_outputs", False)),
            "restart_arg": metadata.get("restart_arg"),
            "parallel_group": metadata.get("parallel_group"),
        })
    return stages


def _stage_batches(stages):
    """Agrupa solo comandos contiguos con el mismo grupo, respetando dependencias."""
    batches = []
    index = 0
    while index < len(stages):
        first = stages[index]
        group = first.get("parallel_group")
        if not group:
            batches.append([first])
            index += 1
            continue

        batch = [first]
        batch_ids = {first["id"]}
        cursor = index + 1
        while cursor < len(stages):
            candidate = stages[cursor]
            if candidate.get("parallel_group") != group:
                break
            # Si un comando depende de otro dentro del lote, inicia un lote nuevo.
            if batch_ids.intersection(candidate.get("depends_on", [])):
                break
            batch.append(candidate)
            batch_ids.add(candidate["id"])
            cursor += 1
        batches.append(batch)
        index = cursor
    return batches


def _run_stage(stage, state, log_dir, overwrite=False):
    stage_id = stage["id"]
    existing_outputs = [path for path in stage["outputs"] if Path(path).exists()]
    restart = bool(stage.get("restart_arg") and stage["command"] and stage["restart_arg"] in stage["command"])
    if not overwrite and not restart and existing_outputs:
        if all(_file_is_valid(path) for path in stage["outputs"]):
            print(f"[{stage_id}] salidas existentes verificadas; se omite.", flush=True)
            with STATE_LOCK:
                state["stages"][stage_id] = {
                    "status": "done", "finished_at": _now(), "skipped_existing": True,
                }
                _write_state(state)
            return True

        if stage.get("allow_partial_outputs"):
            print(f"[{stage_id}] hay salidas parciales; se ejecutará para reanudar o completar.", flush=True)
        else:
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
        running_stages = state.setdefault("running_stages", [])
        if stage_id not in running_stages:
            running_stages.append(stage_id)
        state["current_stages"] = list(running_stages)
        state["current_stage"] = running_stages[0]
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
                        running_stages = state.get("running_stages", [])
                        if stage_id in running_stages:
                            running_stages.remove(stage_id)
                        state["current_stages"] = list(running_stages)
                        state["current_stage"] = running_stages[0] if running_stages else None
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
        running_stages = state.get("running_stages", [])
        if stage_id in running_stages:
            running_stages.remove(stage_id)
        state["current_stages"] = list(running_stages)
        state["current_stage"] = running_stages[0] if running_stages else None
        _write_state(state)
    if return_code != 0:
        print(
            f"[{stage_id}] fallo con codigo {return_code}; revisa {_display_path(log_path)}",
            flush=True,
        )
        return False
    print(f"[{stage_id}] terminado.", flush=True)
    return True


def _run_pipeline(*, workers=1, sgd_jobs=1, full_evaluation=False, include_contextual=True, notify=True, overwrite=False, log_dir=None, plan_path=None):
    log_dir = Path(log_dir) if log_dir else config.PIPELINE_LOGS
    log_dir.mkdir(parents=True, exist_ok=True)
    config.STOP_FILE.unlink(missing_ok=True)
    state = _read_state()
    plan = _load_plan(plan_path)
    stages = _stages(plan, sgd_jobs, full_evaluation, include_contextual, overwrite)
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
        "plan": plan,
        "plan_path": str(Path(plan_path or config.PIPELINE_PLAN).resolve()),
        "stages": {stage["id"]: {"status": "pending"} for stage in stages},
    })
    _write_state(state)
    success = True
    try:
        for batch in _stage_batches(stages):
            if workers == 1 or len(batch) == 1:
                outcomes = [_run_stage(batch[0], state, log_dir, overwrite)]
            else:
                with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as pool:
                    outcomes = list(pool.map(
                        lambda stage: _run_stage(stage, state, log_dir, overwrite),
                        batch,
                    ))
            if not all(outcomes):
                success = False
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
        state.pop("current_stages", None)
        state.pop("running_stages", None)
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

    plan_path = Path(args.plan or config.PIPELINE_PLAN).resolve()
    plan = _load_plan(plan_path)
    _stages(plan, args.sgd_jobs, args.full_evaluation, not args.skip_contextual, args.overwrite)
    log_dir = _clear_schedule_artifacts()
    log_path = log_dir / "scheduler.log"
    command = _poetry_command(
        "schedule", "_worker",
        "--log-dir", str(log_dir),
        "--workers", str(args.workers),
        "--sgd-jobs", str(args.sgd_jobs),
        "--plan", str(plan_path),
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
        try:
            stages = _stages(
                state.get("plan") or _load_plan(state.get("plan_path")),
                state.get("sgd_jobs", 1),
                state.get("full_evaluation", False),
                state.get("include_contextual", True),
                state.get("overwrite", False),
            )
        except (ValueError, OSError):
            stages = []
        for stage in stages:
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
    current_stages = state.get("current_stages") or ([state["current_stage"]] if state.get("current_stage") else [])
    print(f"Etapas activas: {', '.join(current_stages) if current_stages else '-'}")
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
        from tkinter import messagebox, scrolledtext, simpledialog, ttk
    except ImportError as error:
        print(f"No se pudo cargar Tkinter: {_console_text(error)}", file=sys.stderr)
        return 1

    root = tk.Tk()
    root.title("Pipeline — estado de ejecución")
    root.geometry("1080x820")
    root.minsize(820, 620)

    status_text = tk.StringVar(value="Consultando estado…")
    current_stage_text = tk.StringVar(value="Etapa activa: —")
    overwrite = tk.BooleanVar(value=False)
    full_evaluation = tk.BooleanVar(value=False)
    workers = tk.IntVar(value=1)

    try:
        saved_plan = _load_plan()
    except ValueError as error:
        messagebox.showerror("Plan inválido", str(error), parent=root)
        saved_plan = _default_plan()
    plan_items = {item["name"]: {"name": item["name"], "args": list(item.get("args", []))} for item in saved_plan["commands"]}
    selected_order = [item["name"] for item in saved_plan["commands"]]
    catalog = _load_catalog()
    available_names = [
        name for name, info in catalog.items()
        if info.get("status") == "active" and isinstance(info.get("schedule"), dict)
    ]

    header = ttk.Frame(root, padding=12)
    header.pack(fill="x")
    ttk.Label(header, textvariable=status_text, font=("Segoe UI", 12, "bold")).pack(anchor="w")
    ttk.Label(header, textvariable=current_stage_text).pack(anchor="w", pady=(4, 0))

    plan_frame = ttk.LabelFrame(root, text="Plan de comandos — marca y arrastra para ordenar", padding=8)
    plan_frame.pack(fill="both", expand=False, padx=12, pady=4)
    plan_table = ttk.Treeview(
        plan_frame, columns=("enabled", "order", "name", "args"),
        show="headings", height=9, selectmode="browse",
    )
    for column, heading in (("enabled", "Usar"), ("order", "Orden"), ("name", "Comando"), ("args", "Argumentos (clic para editar)")):
        plan_table.heading(column, text=heading)
    plan_table.column("enabled", width=55, anchor="center", stretch=False)
    plan_table.column("order", width=55, anchor="center", stretch=False)
    plan_table.column("name", width=180, stretch=False)
    plan_table.column("args", width=620)
    plan_table.pack(fill="both", expand=True)
    ttk.Label(plan_frame, text="Las dependencias deben estar antes en la cola, o sus salidas deben existir y ser válidas.").pack(anchor="w", pady=(5, 0))

    def render_plan():
        plan_table.delete(*plan_table.get_children())
        ordered = [name for name in selected_order if name in plan_items]
        ordered.extend(name for name in available_names if name not in ordered)
        selected_positions = {name: index + 1 for index, name in enumerate(selected_order)}
        for name in ordered:
            item = plan_items.get(name, {"name": name, "args": list(catalog[name]["schedule"].get("default_args", []))})
            plan_items[name] = item
            enabled = name in selected_positions
            plan_table.insert(
                "", "end", iid=name,
                values=("☑" if enabled else "☐", selected_positions.get(name, ""), name, json.dumps(item["args"], ensure_ascii=False)),
            )

    render_plan()

    drag_state = {"item": None, "y": None}

    def on_plan_press(event):
        row = plan_table.identify_row(event.y)
        drag_state.update(item=row or None, y=event.y)

    def on_plan_release(event):
        row = plan_table.identify_row(event.y)
        column = plan_table.identify_column(event.x)
        dragged = drag_state.get("item")
        drag_state.update(item=None, y=None)
        if not row or row not in plan_items:
            return
        if dragged == row and column == "#4":
            item = plan_items[row]
            initial = json.dumps(item["args"], ensure_ascii=False)
            value = simpledialog.askstring(
                "Argumentos del comando",
                f"{row}: ingresa una lista JSON de argumentos (ej.: [\"--limit\", \"100\"])",
                initialvalue=initial,
                parent=root,
            )
            if value is not None:
                try:
                    parsed = json.loads(value)
                    if not isinstance(parsed, list) or not all(isinstance(arg, str) for arg in parsed):
                        raise ValueError("Los argumentos deben ser una lista JSON de textos.")
                    item["args"] = parsed
                except (json.JSONDecodeError, ValueError) as error:
                    messagebox.showerror("Argumentos inválidos", str(error), parent=root)
                render_plan()
            return "break"
        if dragged == row and column == "#1":
            if row in selected_order:
                selected_order.remove(row)
            else:
                selected_order.append(row)
            render_plan()
            return "break"
        if dragged and dragged in selected_order and row in selected_order and dragged != row:
            old_index = selected_order.index(dragged)
            new_index = selected_order.index(row)
            selected_order.pop(old_index)
            selected_order.insert(new_index, dragged)
            render_plan()
        return "break"

    plan_table.bind("<ButtonPress-1>", on_plan_press, add=True)
    plan_table.bind("<ButtonRelease-1>", on_plan_release, add=True)

    def current_plan():
        return {"version": 1, "commands": [plan_items[name] for name in selected_order]}

    table = ttk.Treeview(root, columns=("stage", "status", "started", "finished"), show="headings", height=7)
    table.heading("stage", text="Comando")
    table.heading("status", text="Estado")
    table.heading("started", text="Inicio")
    table.heading("finished", text="Fin")
    table.column("stage", width=180)
    table.column("status", width=120, anchor="center")
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
    ttk.Checkbutton(controls, text="Sobrescribir salidas", variable=overwrite).pack(side="left")
    ttk.Checkbutton(controls, text="Evaluación SGD completa", variable=full_evaluation).pack(side="left", padx=(10, 0))
    ttk.Label(controls, text="Workers por grupo:").pack(side="left", padx=(12, 4))
    ttk.Spinbox(controls, from_=1, to=32, width=4, textvariable=workers).pack(side="left")

    def save_plan():
        try:
            plan = current_plan()
            _stages(plan)
            _save_plan(plan)
            status_text.set(f"Plan guardado en {_display_path(config.PIPELINE_PLAN)}")
        except (ValueError, OSError) as error:
            messagebox.showerror("No se pudo guardar el plan", str(error), parent=root)

    ttk.Button(controls, text="Guardar plan", command=save_plan).pack(side="right", padx=(8, 0))

    def start_pipeline():
        nonlocal active_log_path, log_offset
        options = argparse.Namespace(
            workers=workers.get(),
            sgd_jobs=1,
            full_evaluation=full_evaluation.get(),
            skip_contextual=False,
            no_notify=False,
            overwrite=overwrite.get(),
            plan=str(config.PIPELINE_PLAN),
        )
        active_log_path = None
        log_offset = 0
        log_view.configure(state="normal")
        log_view.delete("1.0", "end")
        log_view.configure(state="disabled")
        try:
            _save_plan(current_plan())
            _start_background(options)
        except (OSError, RuntimeError, ValueError) as error:
            messagebox.showerror("No se pudo iniciar el schedule", str(error), parent=root)

    ttk.Button(controls, text="Iniciar", command=start_pipeline).pack(side="right", padx=(8, 0))
    ttk.Button(controls, text="Detener", command=_stop).pack(side="right")

    def refresh():
        nonlocal active_log_path, log_offset
        state = _read_state()
        current_status = state.get("status", "sin ejecuciones")
        if current_status == "completed":
            try:
                stages = _stages(
                    state.get("plan") or _load_plan(state.get("plan_path")),
                    state.get("sgd_jobs", 1),
                    state.get("full_evaluation", False),
                    state.get("include_contextual", True),
                    state.get("overwrite", False),
                )
            except (ValueError, OSError):
                stages = []
            for stage in stages:
                recorded = state.get("stages", {}).get(stage["id"], {})
                if recorded.get("status") == "done" and not all(_file_is_valid(path) for path in stage["outputs"]):
                    current_status = "completed_inconsistent; faltan salidas"
                    break
        status_text.set(f"Estado: {current_status} | PID: {state.get('pid', '—')}")
        active_stages = state.get("current_stages") or ([state["current_stage"]] if state.get("current_stage") else [])
        current_stage_text.set(f"Etapas activas: {', '.join(active_stages) if active_stages else '—'}")
        stage_records = state.get("stages", {})
        stage_order = [item["name"] for item in state.get("plan", {}).get("commands", [])]
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
        command.add_argument("--workers", type=int, default=1, help="Máximo de comandos simultáneos dentro de un grupo consecutivo (default: 1).")
        command.add_argument("--sgd-jobs", type=int, default=1, help="Procesos usados por curvas SGD (default: 1).")
        command.add_argument("--full-evaluation", action="store_true", help="SGD usa 5 folds y 5 puntos de curva; mas lento.")
        command.add_argument("--skip-contextual", action="store_true", help="Omite BERT y ELMo, que no son requeridos por SGD.")
        command.add_argument("--no-notify", action="store_true", help="Desactiva el aviso del area de notificacion al terminar.")
        command.add_argument(
            "--overwrite", action="store_true",
            help="Regenera las salidas existentes; por defecto se conservan y se omiten.",
        )
        command.add_argument("--plan", type=Path, default=config.PIPELINE_PLAN, help="Archivo JSON del plan (default: optimization_results/plan.json).")

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
    worker.add_argument("--plan", type=Path, default=config.PIPELINE_PLAN)
    args = parser.parse_args(argv)

    if args.action in {"start", "run", "_worker"}:
        if args.workers < 1 or args.sgd_jobs < 1:
            parser.error("workers y sgd-jobs deben ser enteros positivos")
        if args.action == "start":
            try:
                return _start_background(args)
            except (OSError, ValueError) as error:
                parser.error(str(error))
        if args.action == "run":
            try:
                plan = _load_plan(args.plan)
                _stages(plan, args.sgd_jobs, args.full_evaluation, not args.skip_contextual, args.overwrite)
            except (OSError, ValueError) as error:
                parser.error(str(error))
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
            plan_path=args.plan,
        )
    if args.action == "status":
        return _status()
    if args.action == "plan":
        try:
            plan = _load_plan()
        except ValueError as error:
            parser.error(str(error))
        for number, item in enumerate(plan["commands"], start=1):
            args_text = f" {json.dumps(item.get('args', []), ensure_ascii=False)}" if item.get("args") else ""
            print(f"{number}. {item['name']}{args_text}")
        return 0
    if args.action == "gui":
        return _gui()
    return _stop()


if __name__ == "__main__":
    raise SystemExit(main())
