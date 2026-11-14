import argparse
import json
import re
import sys
import tomlkit
import subprocess

from config import JSON_COMMANDS, TOML_PATH, PROJECT_ROOT

COMMAND_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")
ENTRYPOINT = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*:[a-zA-Z_][a-zA-Z0-9_]*$")

VALID_STATUSES = {"active", "deprecated", "retired"}

def load_catalog():
    with JSON_COMMANDS.open(encoding="utf-8") as file:
        return json.load(file)


def sync_project(*, install=True):

    catalog = load_catalog()
    original_toml = TOML_PATH.read_text(encoding="utf-8")
    project = tomlkit.parse(original_toml)

    project_section = project["project"]
    scripts = project_section.setdefault("scripts", tomlkit.table())

    desired_scripts = {
        name: info["entrypoint"]
        for name, info in catalog["commands"].items()
        if info.get("status") in {"active", "deprecated"}
        and info.get("entrypoint")
    }
    scripts.clear()
    for name, entrypoint in desired_scripts.items():
        scripts[name] = entrypoint

    # utilidades compartidas importadas por varios comandos; no son entrypoints
    # independientes, por lo que no aparecen en commands.info.json.
    includes = {"config.py", "utils.py"}
    for info in catalog["commands"].values():
        include = info.get("include")
        if include:
            includes.add(include.strip().replace("\\", "/"))

    packages = tomlkit.array()
    packages.multiline(True)
    for include in sorted(includes):
        package = tomlkit.inline_table()
        package["include"] = include
        package["from"] = "src"
        packages.append(package)

    poetry_section = project.setdefault("tool", tomlkit.table()).setdefault(
        "poetry", tomlkit.table()
    )
    poetry_section["packages"] = packages

    updated_toml = tomlkit.dumps(project)
    changed = updated_toml != original_toml

    if not changed:
        print("El proyecto ya esta sincronizado.")
        return 0

    try:
        TOML_PATH.write_text(updated_toml, encoding="utf-8")
        if install:
            result = subprocess.run(
                ["poetry", "install"],
                cwd=PROJECT_ROOT,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"poetry install failed with exit code {result.returncode}"
                )
    except Exception as error:
        TOML_PATH.write_text(original_toml, encoding="utf-8")
        if install:
            subprocess.run(
                ["poetry", "install"],
                cwd=PROJECT_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        print(f"La sincronizacion ha fallado: {error}", file=sys.stderr)
        print("pyproject.toml fue restaurado.", file=sys.stderr)
        return 1

    print("Proyecto sincronizado correctamente.")
    return 0


def save_and_sync(catalog):
    original_json = JSON_COMMANDS.read_text(encoding="utf-8")
    JSON_COMMANDS.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = sync_project()
    if result != 0:
        JSON_COMMANDS.write_text(original_json, encoding="utf-8")
        sync_project()
        return 1
    return 0


def list_tools(show_all=False):
    catalog = load_catalog()
    print("Herramientas disponibles del proyecto\n")

    for name, info in catalog["commands"].items():
        if not show_all and info.get("status") != "active":
            continue

        print(f"{name} — {info.get('description', 'No description')}")
        print(f"  Entrypoint: {info.get('entrypoint', 'Not configured')}")
        print(f"  Include: {info.get('include', 'Not configured')}")
        print(f"  Status: {info.get('status', 'unknown')}")

        if info.get("supports_help", False):
            print(f"  Parameters: Yes — run `poetry run {name} --help`")
        else:
            print("  Parameters: No")

        print()

def prompt_value(label, current=None, *, required=False):
    suffix = f" [{current}]" if current else ""
    value = input(f"{label}{suffix}: ").strip()
    if not value and current is not None:
        return current
    if required and not value:
        raise ValueError(f"{label} no puede estar vacio.")
    return value


def prompt_bool(label, current=None):
    suffix = " (true/false)"
    if current is not None:
        suffix = f" (true/false) [{str(current).lower()}]"

    while True:
        value = input(f"{label}{suffix}: ").strip().lower()

        if not value and current is not None:
            return current

        if value in {"true", "t", "yes", "y", "si", "sí", "1"}:
            return True

        if value in {"false", "f", "no", "n", "0"}:
            return False

        print("Valor inválido. Ingrese true o false.")


def prompt_json_list(label, current=None):
    """Solicita una lista JSON, útil para argumentos con espacios o comas."""
    default = json.dumps(current or [], ensure_ascii=False)
    while True:
        value = input(f"{label} (JSON) [{default}]: ").strip()
        if not value:
            return current or []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            print(f"JSON inválido: {error}")
            continue
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed
        print("Ingrese una lista JSON cuyos elementos sean textos.")


def prompt_json_object(label, current=None):
    default = json.dumps(current or {}, ensure_ascii=False)
    while True:
        value = input(f"{label} (JSON) [{default}]: ").strip()
        if not value:
            return current or {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            print(f"JSON inválido: {error}")
            continue
        if isinstance(parsed, dict):
            return parsed
        print("Ingrese un objeto JSON.")


def prompt_schedule(current=None):
    """Solicita la metadata que permite incluir un comando en un plan."""
    current = current or {}
    enabled = prompt_bool(
        "¿Incluir este comando en los planes de schedule?",
        bool(current),
    )
    if not enabled:
        return None

    print("Use rutas relativas a la raíz del proyecto para las salidas y nombres de comandos para dependencias.")
    outputs = prompt_json_list("Salidas (p. ej. [\"data/dataset_clean.csv\"])", current.get("outputs", []))
    depends_on = prompt_json_list("Depende de (nombres de comandos)", current.get("depends_on", []))
    parallel_group = prompt_value(
        "Grupo paralelo (vacío = ejecución secuencial)",
        current.get("parallel_group", ""),
    )
    default_args = prompt_json_list(
        "Argumentos por defecto (p. ej. [\"--download-model\"])",
        current.get("default_args", []),
    )
    overwrite_arg = prompt_value(
        "Argumento para permitir sobrescritura (vacío si no admite)",
        current.get("overwrite_arg", "") or "",
    )
    outputs_by_argument = prompt_json_object(
        "Salidas condicionales por opción (objeto option -> valor -> rutas)",
        current.get("outputs_by_argument", {}),
    )
    depends_on_by_argument = prompt_json_object(
        "Dependencias condicionales por opción (objeto option -> valor -> comandos)",
        current.get("depends_on_by_argument", {}),
    )
    allow_partial_outputs = prompt_bool(
        "¿Permitir ejecutar con salidas parciales para reanudar?",
        current.get("allow_partial_outputs", False),
    )
    restart_arg = prompt_value(
        "Argumento que fuerza reinicio (vacío si no aplica)", current.get("restart_arg", "") or ""
    )
    return {
        "outputs": outputs,
        "depends_on": depends_on,
        "parallel_group": parallel_group or None,
        "default_args": default_args,
        "overwrite_arg": overwrite_arg or None,
        "outputs_by_argument": outputs_by_argument,
        "depends_on_by_argument": depends_on_by_argument,
        "allow_partial_outputs": allow_partial_outputs,
        "restart_arg": restart_arg or None,
    }


def validate_command(name, entrypoint, description, include, catalog, *, old_name=None, schedule=None):
    if not COMMAND_NAME.fullmatch(name):
        raise ValueError("Nombre del comando invalido.")
    if not ENTRYPOINT.fullmatch(entrypoint):
        raise ValueError("Entrypoint debe utilizar el formato de module:function.")
    if not description.strip():
        raise ValueError("La descripcion no puede estar vacia.")
    if not include.strip():
        raise ValueError("Include no puede estar vacia; use un path relativo a src/.")
    if old_name is None and name in catalog["commands"]:
        raise ValueError(f"'{name}' ya existe en el catalogo.")
    if old_name is not None and name != old_name and name in catalog["commands"]:
        raise ValueError(f"'{name}' ya existe en el catalogo.")
    if schedule is not None:
        if not isinstance(schedule, dict):
            raise ValueError("La configuración schedule debe ser un objeto o null.")
        for field in ("outputs", "depends_on", "default_args"):
            values = schedule.get(field, [])
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ValueError(f"schedule.{field} debe ser una lista de textos.")
        if not isinstance(schedule.get("parallel_group"), (str, type(None))):
            raise ValueError("schedule.parallel_group debe ser un texto o null.")
        if not isinstance(schedule.get("overwrite_arg"), (str, type(None))):
            raise ValueError("schedule.overwrite_arg debe ser un texto o null.")
        if not isinstance(schedule.get("restart_arg"), (str, type(None))):
            raise ValueError("schedule.restart_arg debe ser un texto o null.")
        if not isinstance(schedule.get("allow_partial_outputs", False), bool):
            raise ValueError("schedule.allow_partial_outputs debe ser booleano.")
        for output in schedule.get("outputs", []):
            normalized = output.replace("\\", "/")
            if normalized.startswith("/") or ".." in normalized.split("/"):
                raise ValueError("Las salidas schedule deben ser rutas relativas dentro del proyecto.")
        for option, choices in schedule.get("outputs_by_argument", {}).items():
            if not isinstance(option, str) or not isinstance(choices, dict):
                raise ValueError("schedule.outputs_by_argument debe mapear opciones a objetos valor -> rutas.")
            for paths in choices.values():
                if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
                    raise ValueError("Cada salida condicional debe ser una lista de rutas.")
                for output in paths:
                    normalized = output.replace("\\", "/")
                    if normalized.startswith("/") or ".." in normalized.split("/"):
                        raise ValueError("Las salidas schedule deben ser rutas relativas dentro del proyecto.")
        known_commands = set(catalog["commands"])
        if old_name is not None:
            known_commands.discard(old_name)
            known_commands.add(name)
        for dependency in schedule.get("depends_on", []):
            if dependency not in known_commands or dependency == name:
                raise ValueError(f"Dependencia schedule desconocida o circular: '{dependency}'.")
        for option, choices in schedule.get("depends_on_by_argument", {}).items():
            if not isinstance(option, str) or not isinstance(choices, dict):
                raise ValueError("schedule.depends_on_by_argument debe mapear opciones a objetos valor -> comandos.")
            for dependencies in choices.values():
                if not isinstance(dependencies, list) or not all(isinstance(dep, str) for dep in dependencies):
                    raise ValueError("Cada dependencia condicional debe ser una lista de comandos.")
                for dependency in dependencies:
                    if dependency not in known_commands or dependency == name:
                        raise ValueError(f"Dependencia schedule desconocida o circular: '{dependency}'.")


def add_tool(include_arg=None):
    catalog = load_catalog()
    print("Registrar una nueva herramienta del proyecto\n")

    try:
        name = prompt_value("Command name", required=True)
        entrypoint = prompt_value("Entrypoint (module:function)", required=True)
        description = prompt_value("Description", required=True)
        include = include_arg or prompt_value(
            "Include path relativo a src (e.g. analysis or analysis.py)",
            required=True,
        )
        supports_help = prompt_bool("¿Acepta --help?")
        schedule = prompt_schedule()
        validate_command(name, entrypoint, description, include, catalog, schedule=schedule)

    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    catalog["commands"][name] = {
        "entrypoint": entrypoint,
        "description": description,
        "status": "active",
        "include": include.replace("\\", "/").strip(),
        "supports_help": supports_help,
        "schedule": schedule,
    }

    if save_and_sync(catalog) != 0:
        return 1

    print(f"\nTool '{name}' registrada correctamente.")
    print(f"ejecucion disponible con: poetry run {name}")
    return 0


def modify_tool(name, include_arg=None):
    catalog = load_catalog()
    info = catalog["commands"].get(name)

    if info is None:
        print(f"Error: '{name}' no existe en el catalogo.", file=sys.stderr)
        return 1

    print(
        f"Modify project tool: {name}\n"
        "(Presiona enter para conservar el valor actual.)\n"
    )

    try:
        new_name = prompt_value("Command name", name, required=True)
        entrypoint = prompt_value(
            "Entrypoint (module:function)",
            info.get("entrypoint"),
            required=True,
        )
        description = prompt_value(
            "Description",
            info.get("description"),
            required=True,
        )
        include = include_arg or prompt_value(
            "Include path relativo a src",
            info.get("include"),
            required=True,
        )
        status = prompt_value(
            "Status (active/deprecated/retired)",
            info.get("status", "active"),
            required=True,
        ).lower()

        supports_help = prompt_bool(
            "¿Acepta --help?",
            info.get("supports_help", False),
        )
        schedule = prompt_schedule(info.get("schedule"))

        if status not in VALID_STATUSES:
            raise ValueError(
                "Status debe ser 'active', 'deprecated' o 'retired'."
            )

        validate_command(
            new_name,
            entrypoint,
            description,
            include,
            catalog,
            old_name=name,
            schedule=schedule,
        )

    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if new_name != name:
        del catalog["commands"][name]
        for command_info in catalog["commands"].values():
            command_schedule = command_info.get("schedule")
            if command_schedule:
                command_schedule["depends_on"] = [
                    new_name if dependency == name else dependency
                    for dependency in command_schedule.get("depends_on", [])
                ]
                for choices in command_schedule.get("depends_on_by_argument", {}).values():
                    for value, dependencies in choices.items():
                        choices[value] = [new_name if dependency == name else dependency for dependency in dependencies]

    catalog["commands"][new_name] = {
        "entrypoint": entrypoint,
        "description": description,
        "status": status,
        "include": include.replace("\\", "/").strip(),
        "supports_help": supports_help,
        "schedule": schedule,
    }

    if save_and_sync(catalog) != 0:
        return 1

    print(f"Tool '{new_name}' fue actualizada correctamente.")
    return 0


def remove_tool(name):
    catalog = load_catalog()
    info = catalog["commands"].get(name)
    if info is None:
        print(f"Error: '{name}' no existe en el catalogo.", file=sys.stderr)
        return 1
    if info.get("status") == "retired":
        print(f"Tool '{name}' ya fue eliminada.")
        return 0

    info["status"] = "retired"
    if save_and_sync(catalog) != 0:
        return 1
    print(f"Tool '{name}' marcada como retirada. Su historial permanecera en el catalogo.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Listar y gestionar herramientas CLI del proyecto."
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--add", action="store_true", help="Incluir un nuevo comando.")
    actions.add_argument("--modify", metavar="NAME", help="Modificar un comando existente.")
    actions.add_argument("--remove", metavar="NAME", help="Eliminar un comando sin eliminar su historial.")
    actions.add_argument("--sync", action="store_true", help="Sincronizar la configuración de Poetry desde el catalogo JSON.")
    parser.add_argument("--include", help="Incluir ruta relativa a src/ (e.g. paquete o module.py); usado con --add o --modify.")
    parser.add_argument("--all", action="store_true", help="Listar todos los comandos, incluyendo los obsoletos y retirados.")
    args = parser.parse_args()

    if args.include and not (args.add or args.modify):
        parser.error("--include unicamente puedde ser usado junto a --add o --modify.")
    if args.add:
        raise SystemExit(add_tool(args.include))
    if args.modify:
        raise SystemExit(modify_tool(args.modify, args.include))
    if args.remove:
        raise SystemExit(remove_tool(args.remove))
    if args.sync:
        raise SystemExit(sync_project())
    list_tools(show_all=args.all)

if __name__ == "__main__":
    main()
