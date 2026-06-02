import pathlib

import pytest
from typing import Generator


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch) -> Generator[pathlib.Path, None, None]:
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"
