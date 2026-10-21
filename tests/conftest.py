"""Shared pytest configuration and custom markers."""

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (run with '-m slow')")
