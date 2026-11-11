import queue
import threading

import config

from .client import DriveAccessError
from .sync import DriveDownloader


def open_gui(folder_url=None):
    import tkinter as tk
    from tkinter import messagebox, ttk

    if folder_url is None:
        folder_url = config.DRIVE_FOLDER_URL

    root = tk.Tk()
    root.title("Drive — gestor de descargas")
    root.geometry("980x680")
    root.minsize(760, 500)

    events = queue.Queue()
    downloader = DriveDownloader(folder_url)
    state = {"busy": False, "rows": {}, "result": None}
    status_text = tk.StringVar(value="Preparado para listar la carpeta pública.")
    progress_text = tk.StringVar(value="")

    header = ttk.Frame(root, padding=12)
    header.pack(fill="x")
    ttk.Label(header, text="Carpeta pública de Drive", font=("Segoe UI", 12, "bold")).pack(anchor="w")
    ttk.Label(header, text=folder_url, wraplength=920).pack(anchor="w", pady=(3, 8))
    ttk.Label(header, textvariable=status_text).pack(anchor="w")
    ttk.Label(header, textvariable=progress_text).pack(anchor="w", pady=(2, 0))

    table = ttk.Treeview(
        root,
        columns=("select", "name", "status", "size", "sha", "modified"),
        show="headings",
        height=18,
    )
    for column, title, width in (
        ("select", "Bajar", 55),
        ("name", "Archivo", 300),
        ("status", "Comparación", 130),
        ("size", "Tamaño", 110),
        ("sha", "SHA-256", 145),
        ("modified", "Modificado", 190),
    ):
        table.heading(column, text=title)
        table.column(column, width=width, anchor="center" if column in {"select", "status"} else "w")
    table.pack(fill="both", expand=True, padx=12, pady=8)

    controls = ttk.Frame(root, padding=(12, 0, 12, 12))
    controls.pack(fill="x")
    refresh_button = ttk.Button(controls, text="Actualizar listado")
    mark_button = ttk.Button(controls, text="Marcar diferencias")
    download_button = ttk.Button(controls, text="Descargar seleccionados")
    refresh_button.pack(side="left")
    mark_button.pack(side="left", padx=(8, 0))
    download_button.pack(side="right")

    def set_busy(busy, message=None):
        state["busy"] = busy
        button_state = "disabled" if busy else "normal"
        refresh_button.configure(state=button_state)
        mark_button.configure(state=button_state)
        download_button.configure(state=button_state)
        if message:
            status_text.set(message)

    def refresh_list():
        if state["busy"]:
            return
        set_busy(True, "Listando Drive, guardando ListadoDescarga.json y leyendo ListadoRemoto.json…")
        progress_text.set("")

        def work():
            try:
                result = downloader.scan(
                    progress=lambda name, size, total: events.put(("progress", name, size, total))
                )
                events.put(("scan_done", result))
            except Exception as error:
                events.put(("error", str(error)))

        threading.Thread(target=work, daemon=True).start()

    def toggle_selection(event):
        if state["busy"] or table.identify_column(event.x) != "#1":
            return
        item_id = table.identify_row(event.y)
        if item_id not in state["rows"]:
            return
        values = list(table.item(item_id, "values"))
        values[0] = "☑" if values[0] == "☐" else "☐"
        table.item(item_id, values=values)
    table.bind("<Button-1>", toggle_selection)

    def mark_differences():
        if state["busy"]:
            return
        for drive_id, compared in state["rows"].items():
            values = list(table.item(drive_id, "values"))
            values[0] = "☑" if compared.status != "igual" else "☐"
            table.item(drive_id, values=values)

    def download_selected():
        if state["busy"] or state["result"] is None:
            return
        selected = {
            drive_id
            for drive_id in state["rows"]
            if table.set(drive_id, "select") == "☑"
        }
        if not selected:
            messagebox.showinfo("Sin selección", "Marca los archivos que quieres descargar.", parent=root)
            return
        set_busy(True, f"Descargando {len(selected)} archivo(s)…")

        def work():
            try:
                downloaded, failures = downloader.download(
                    selected_ids=selected,
                    progress=lambda name, size, total: events.put(("progress", name, size, total)),
                )
                events.put(("download_done", downloaded, failures))
            except Exception as error:
                events.put(("error", str(error)))

        threading.Thread(target=work, daemon=True).start()

    def render_result(result):
        table.delete(*table.get_children())
        state["rows"] = {}
        for compared in result.files:
            item = compared.metadata
            drive_id = compared.drive_file.drive_id
            state["rows"][drive_id] = compared
            size = f"{item['size_bytes']:,} bytes"
            table.insert(
                "", "end", iid=drive_id,
                values=("☐", item["name"], compared.status, size, item["sha256"][:16], item["modified_at"]),
            )
        for index, name in enumerate(result.local_only):
            iid = f"local-only-{index}"
            table.insert("", "end", iid=iid, values=("—", name, "solo local", "—", "—", "—"))
        extras = len(result.untracked_remote)
        summary = f"{len(result.files)} archivo(s) del manifiesto; {len(result.local_only)} solo local."
        if extras:
            summary += f" {extras} archivo(s) remoto(s) no incluidos en ListadoRemoto.json."
        status_text.set(summary)

    def poll_events():
        try:
            while True:
                event = events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    _, name, size, total = event
                    if total:
                        progress_text.set(f"{name}: {size:,} / {total:,} bytes")
                    else:
                        progress_text.set(f"Recibiendo {name}: {size:,} bytes")
                elif kind == "scan_done":
                    state["result"] = event[1]
                    render_result(event[1])
                    progress_text.set("")
                    set_busy(False)
                elif kind == "download_done":
                    _, downloaded, failures = event
                    progress_text.set("")
                    message = f"Descargados: {len(downloaded)}"
                    if failures:
                        message += f"; fallidos: {len(failures)}"
                        messagebox.showerror(
                            "Descargas incompletas",
                            "\n".join(f"{name}: {error}" for name, error in failures),
                            parent=root,
                        )
                    status_text.set(message)
                    set_busy(False)
                    if not failures:
                        refresh_list()
                elif kind == "error":
                    progress_text.set("")
                    set_busy(False, "La operación falló.")
                    messagebox.showerror("Error de Drive", event[1], parent=root)
        except queue.Empty:
            pass
        root.after(150, poll_events)

    def close_window():
        if state["busy"]:
            messagebox.showwarning("Operación en curso", "Espera a que termine la operación.", parent=root)
            return
        downloader.close()
        root.destroy()

    refresh_button.configure(command=refresh_list)
    mark_button.configure(command=mark_differences)
    download_button.configure(command=download_selected)
    root.protocol("WM_DELETE_WINDOW", close_window)
    poll_events()
    root.after(100, refresh_list)
    root.mainloop()
    return 0
