import io

import numpy as np
import pytest

from oddball import Dataset, clear_cache, list_available, load, split_by_label
from oddball.datasets import DatasetManager
from oddball.registry import DATASET_FILES


@pytest.fixture(autouse=True)
def reset_cache():
    clear_cache(all_versions=True)
    yield
    clear_cache(all_versions=True)


@pytest.fixture
def sample_npz_bytes():
    def factory(n_rows: int = 50, n_features: int = 4) -> bytes:
        rng = np.random.default_rng(123)
        X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
        y = np.zeros(n_rows, dtype=np.int64)
        y[: max(1, n_rows // 10)] = 1
        rng.shuffle(y)
        buf = io.BytesIO()
        np.savez(buf, X=X, y=y)
        return buf.getvalue()

    return factory


@pytest.fixture
def mock_download(monkeypatch, sample_npz_bytes):
    payload = sample_npz_bytes()

    def fake_download(self, filename: str) -> bytes:  # noqa: ARG001
        return payload

    monkeypatch.setattr(DatasetManager, "_download", fake_download)
    return payload


def test_load_returns_raw_arrays(mock_download):  # noqa: ARG001
    X, y = load(Dataset.COVER)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]
    assert set(np.unique(y)) <= {0, 1}


def test_list_available_matches_registry():
    assert sorted(list_available()) == sorted(DATASET_FILES.keys())


@pytest.mark.parametrize("name", list_available())
def test_all_datasets_can_load_via_global_loader(mock_download, name):  # noqa: ARG001
    X, y = load(name)
    assert X.shape[0] == y.shape[0] > 0


@pytest.mark.parametrize(
    "alias",
    [
        "page_blocks",
        "page-blocks",
        "internet_ads",
        "spam_base",
        "breast wisconsin",
        "satimage_2",
    ],
)
def test_aliases_resolve_to_supported_dataset(mock_download, alias):  # noqa: ARG001
    X, y = load(alias)
    assert X.shape[0] == y.shape[0] > 0


def test_split_by_label_returns_views(mock_download):  # noqa: ARG001
    normal, anomaly = split_by_label(Dataset.COVER)
    assert isinstance(normal, np.ndarray)
    assert isinstance(anomaly, np.ndarray)
    assert normal.shape[1] == anomaly.shape[1]
