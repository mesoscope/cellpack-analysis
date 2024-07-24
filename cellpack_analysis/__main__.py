import importlib
import sys

from cellpack_analysis.__config__ import (
    display_config,
    make_config_from_dotlist,
    make_config_from_file,
    make_config_from_yaml,
)

def main() -> None:
    if len(sys.argv) < 2:
        return

    if "--dryrun" in sys.argv:
        sys.argv.remove("--dryrun")
        dryrun = True
    else:
        dryrun = False

    module_name = sys.argv[1].replace("-", "_")
    module = get_module(module_name)

    if module is None:
        return

    if len(sys.argv) > 2 and sys.argv[2] == "::":
        config = make_config_from_dotlist(module, sys.argv[3:])
    elif len(sys.argv) == 3:
        config = make_config_from_file(module, sys.argv[2])
    else:
        config = make_config_from_yaml(module, sys.argv[2:])

    display_config(config)

    if dryrun:
        return
    

def get_module(module_name: str) -> None:
    # TODO: add modules based on structure, e.g., recipe generation, packing, etc.
    module_spec = importlib.util.find_spec(f"{module_name}.run_packing_workflow", package=__name__)

    if module_spec is not None:
        module = importlib.import_module(f"{module_name}.run_packing_workflow", package=__name__)
    else:
    #     response = input(f"Module {module_name} does not exist. Create template for module [y/n]? ")
    #     if response[0] == "y":
    #         create_flow_template(module_name)

        module = None

    return module
    

if __name__ == "__main__":
    main()