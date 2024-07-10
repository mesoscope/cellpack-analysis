import importlib.util
import os
import subprocess

from dotenv import load_dotenv

# set cellPACK path
load_dotenv()
CELLPACK_PATH = os.getenv("CELLPACK")
if CELLPACK_PATH is None:
    spec = importlib.util.find_spec("cellpack")
    if spec is None:
        raise Exception("cellPACK not found")
    CELLPACK_PATH = spec.origin

PACK_PATH: str = CELLPACK_PATH + "/cellpack/bin/pack.py"  # type: ignore
assert os.path.exists(PACK_PATH), f"PACK path {PACK_PATH} does not exist"


def run_single_packing(
    recipe_path,
    config_path,
):
    try:
        print(f"Running {recipe_path}")
        result = subprocess.run(
            [
                "python",
                PACK_PATH,
                "-r",
                recipe_path,
                "-c",
                config_path,
            ],
            check=True,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False
