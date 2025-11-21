# oddball

Lightweight access to the 47 ADBench classical anomaly detection datasets.
Downloads the published `.npz` assets from the GitHub release on demand and
returns raw `(X, y)` NumPy arrays.

- Default assets: https://github.com/OliverHennhoefer/oddball/releases/tag/v1.0-datasets
- Cache: `~/.cache/oddball/<version>` (override with env vars below)

## Installation

```bash
pip install oddball
```

## Usage

```python
from oddball import Dataset, load, split_by_label, list_available

print("Available:", list_available())

X, y = load(Dataset.COVER)           # raw arrays
normal, anomaly = split_by_label("cover")  # feature slices
```

## Configuration

- `ODDBALL_DATASET_VERSION` (default: `v1.0-datasets`)
- `ODDBALL_DATASET_URL` (default: `https://github.com/OliverHennhoefer/oddball/releases/download/<version>/`)
- `ODDBALL_CACHE_DIR` (default: `~/.cache/oddball/<version>`)
- `.env` support: place the above keys in `.env` (or set `ODDBALL_DOTENV=/path/to/.env`).

## Supported datasets

All 47 ADBench classical datasets are available. Call `oddball.list_available()` to see slugs (e.g., `cover`, `fraud`, `satimage2`).
