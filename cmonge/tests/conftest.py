from pathlib import Path

import pytest
from cmonge.datasets.single_loader import SciPlexModule
from cmonge.utils import load_config


@pytest.fixture(scope="session")
def synthetic_config():
    config_path = Path("cmonge/tests/configs/synthetic.yml")
    config = load_config(config_path)
    return config


@pytest.fixture(scope="module")
def synthetic_data(synthetic_config):
    module = SciPlexModule(synthetic_config.data)
    return module
