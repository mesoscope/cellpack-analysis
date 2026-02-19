"""cellpack_analysis package - Analysis methods for cellPACK."""

__author__ = "Saurabh Mogre"
__email__ = "saurabh.mogre@alleninstitute.org"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.0.1"

import logging
import logging.config
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_module_version() -> str:
    """Get the current module version."""
    return __version__


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration for the entire package."""
    config_path = Path(__file__).parents[1] / "logging.conf"
    if config_path.exists():
        logging.config.fileConfig(config_path, disable_existing_loggers=False)
        # Don't override the root logger level from logging.conf
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
        logging.getLogger().setLevel(level)


# Initialize logging automatically when the package is imported
setup_logging()


__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "get_module_version",
    "setup_logging",
]
