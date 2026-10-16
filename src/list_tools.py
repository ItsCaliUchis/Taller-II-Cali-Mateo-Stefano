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

    includes = {"config.py"}
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
        print(f"  Status: {info.get('status', 'unknown')}\n")


def prompt_value(label, current=None, *, required=False):
    suffix = f" [{current}]" if current else ""
    value = input(f"{label}{suffix}: ").strip()
    if not value and current is not None:
        return current
    if required and not value:
        raise ValueError(f"{label} no puede estar vacio.")
    return value


def validate_command(name, entrypoint, description, include, catalog, *, old_name=None):
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
        validate_command(name, entrypoint, description, include, catalog)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    catalog["commands"][name] = {
        "entrypoint": entrypoint,
        "description": description,
        "status": "active",
        "include": include.replace("\\", "/").strip(),
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
        print(f"Error: '{name}' No existe en el catalogo.", file=sys.stderr)
        return 1

    print(f"Modify project tool: {name}\n(Presiona enter para conservar el valor actual.)\n")
    try:
        new_name = prompt_value("Command name", name, required=True)
        entrypoint = prompt_value("Entrypoint (module:function)", info.get("entrypoint"), required=True)
        description = prompt_value("Description", info.get("description"), required=True)
        include = include_arg or prompt_value("Include path relativo a src", info.get("include"), required=True)
        status = prompt_value("Status (active/deprecated/retired)", info.get("status", "active"), required=True).lower()
        if status not in VALID_STATUSES:
            raise ValueError("Status debe ser 'active', 'deprecated' o 'retired'.")
        validate_command(new_name, entrypoint, description, include, catalog, old_name=name)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if new_name != name:
        del catalog["commands"][name]
    catalog["commands"][new_name] = {
        "entrypoint": entrypoint,
        "description": description,
        "status": status,
        "include": include.replace("\\", "/").strip(),
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
