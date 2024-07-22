# -*- coding: utf-8 -*-

"""Top-level package for cellpack_analysis."""

__author__ = "Saurabh Mogre"
__email__ = "saurabh.mogre@alleninstitute.org"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.0.1"


from dotenv import load_dotenv


def get_module_version():
    return __version__


load_dotenv()
