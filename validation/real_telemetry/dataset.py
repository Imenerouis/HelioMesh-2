"""
HelioMesh -- Real Telemetry Benchmark Dataset
==============================================
Attempts to download the NASA SMAP/MSL telemetry anomaly dataset
(Hundman et al., 2018).  If the download fails for any reason the
function returns a fully-documented SYNTHETIC_STANDIN that mimics
the same structure.

IMPORTANT -- what this is and what it is NOT:
  - IS:      An independent anomaly-detection benchmark track,
             separate from the HelioMesh simulation states.
  - IS NOT:  A validation of the RF/GB spacecraft-health models.
             Anomaly labels here are NOT mapped to NOMINAL /
             STANDBY / SAFE_MODE / CRITICAL_AHEAD.

Dataset structure returned by load_dataset():
  {
    "source":    "SMAP_REAL" | "SMAP_STANDIN",
    "channels":  ["power_volt", "thermal_temp", "cmd_ack"],
    "n_timesteps": int,
    "data":      np.ndarray, shape (n_timesteps, n_channels),
    "labels":    np.ndarray, shape (n_timesteps,), dtype int  (0=normal, 1=anomaly),
    "anomaly_windows": [(start_idx, end_idx), ...]  inclusive, 0-based
  }
"""

import io
import os
import zipfile
import urllib.request
import numpy as np

# ---------------------------------------------------------------------------
# Public URLs
# ---------------------------------------------------------------------------
_SMAP_ZIP_URL = (
    "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
)
_LABELS_CSV_URL = (
    "https://raw.githubusercontent.com/khundman/telemanom/"
    "master/labeled_anomalies.csv"
)

# Channels to use from the SMAP dataset (sub-set kept to 3 for parity with
# the synthetic stand-in)
_SMAP_CHANNELS = ["P-1", "S-1", "E-1"]

_TIMEOUT_SECS = 15  # network timeout per request


# ---------------------------------------------------------------------------
# SYNTHETIC STAND-IN
# ---------------------------------------------------------------------------
def _make_synthetic_standin(
    n_timesteps: int = 500,
    n_channels: int = 3,
    anomaly_fraction: float = 0.20,
    seed: int = 42,
) -> dict:
    """
    Reproducible synthetic dataset that mimics SMAP structure.

    Anomaly windows are injected at fixed positions covering ~20% of the
    series.  Anomalous segments have elevated mean and variance relative
    to the normal background.

    This dataset is labelled SYNTHETIC_STANDIN.  It is NOT the real SMAP
    dataset.  Results produced with it are clearly marked as such.
    """
    rng = np.random.default_rng(seed)

    channel_names = ["power_volt", "thermal_temp", "cmd_ack"]
    data = np.zeros((n_timesteps, n_channels), dtype=np.float32)
    labels = np.zeros(n_timesteps, dtype=int)

    # Build anomaly windows: 4 non-overlapping blocks, each ~2.5% of series
    block = int(n_timesteps * anomaly_fraction / 4)
    offsets = [50, 150, 280, 410]
    anomaly_windows = []
    for off in offsets:
        start = min(off, n_timesteps - block - 1)
        end = min(start + block - 1, n_timesteps - 1)
        anomaly_windows.append((start, end))
        labels[start : end + 1] = 1

    # Generate per-channel signals
    for ch in range(n_channels):
        base = rng.normal(loc=0.0, scale=1.0, size=n_timesteps).astype(np.float32)
        # inject anomaly spikes
        for start, end in anomaly_windows:
            length = end - start + 1
            base[start : end + 1] += rng.normal(
                loc=3.5 + ch * 0.5, scale=1.2, size=length
            ).astype(np.float32)
        data[:, ch] = base

    return {
        "source": "SMAP_STANDIN",
        "channels": channel_names,
        "n_timesteps": n_timesteps,
        "data": data,
        "labels": labels,
        "anomaly_windows": anomaly_windows,
        "disclaimer": (
            "SYNTHETIC_STANDIN: this is NOT the real NASA SMAP dataset. "
            "Anomaly windows are artificially injected at known positions "
            "(seed=42).  Results are for methodology demonstration only."
        ),
    }


# ---------------------------------------------------------------------------
# REAL SMAP LOADER (best-effort)
# ---------------------------------------------------------------------------
def _load_smap_real() -> dict:
    """
    Download and parse real SMAP test-set channels.
    Raises RuntimeError if anything fails so the caller can fall back.
    """
    import csv

    # -- step 1: fetch labels CSV --
    try:
        with urllib.request.urlopen(_LABELS_CSV_URL, timeout=_TIMEOUT_SECS) as resp:
            csv_text = resp.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(f"labels CSV download failed: {exc}") from exc

    # Parse anomaly windows for the channels we want
    anomaly_map: dict[str, list[tuple[int, int]]] = {}
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        chan = row.get("chan_id", "").strip()
        if chan in _SMAP_CHANNELS:
            seqs = row.get("anomaly_sequences", "[]").strip()
            windows = []
            # format: [[start,end],[start,end],...]
            seqs = seqs.replace("[", "").replace("]", "").strip()
            if seqs:
                pairs = seqs.split(",")
                it = iter(pairs)
                for s, e in zip(it, it):
                    windows.append((int(s.strip()), int(e.strip())))
            anomaly_map[chan] = windows

    # -- step 2: download data zip --
    try:
        with urllib.request.urlopen(_SMAP_ZIP_URL, timeout=_TIMEOUT_SECS) as resp:
            zip_bytes = resp.read()
    except Exception as exc:
        raise RuntimeError(f"data.zip download failed: {exc}") from exc

    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))

    channel_arrays = []
    all_windows: list[tuple[int, int]] = []
    for chan in _SMAP_CHANNELS:
        npy_path = f"data/test/{chan}.npy"
        if npy_path not in zf.namelist():
            raise RuntimeError(f"{npy_path} not found in zip")
        arr = np.load(io.BytesIO(zf.read(npy_path)))  # shape (T, D)
        # Use only the first column (value channel)
        channel_arrays.append(arr[:, 0].astype(np.float32))

    n_ts = min(len(a) for a in channel_arrays)
    data = np.column_stack([a[:n_ts] for a in channel_arrays])

    labels = np.zeros(n_ts, dtype=int)
    final_windows: list[tuple[int, int]] = []
    for chan in _SMAP_CHANNELS:
        for s, e in anomaly_map.get(chan, []):
            s, e = min(s, n_ts - 1), min(e, n_ts - 1)
            labels[s : e + 1] = 1
            final_windows.append((s, e))

    return {
        "source": "SMAP_REAL",
        "channels": _SMAP_CHANNELS,
        "n_timesteps": n_ts,
        "data": data,
        "labels": labels,
        "anomaly_windows": final_windows,
        "disclaimer": (
            "Real NASA SMAP test-set data (Hundman et al., 2018). "
            "Anomaly labels from labeled_anomalies.csv. "
            "NOT a validation of HelioMesh spacecraft health models."
        ),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_dataset() -> dict:
    """
    Return the anomaly benchmark dataset.

    Tries the real SMAP download first; on any failure returns a
    fully-documented synthetic stand-in.
    """
    try:
        print("  [dataset] Attempting SMAP real download...", end=" ", flush=True)
        ds = _load_smap_real()
        print(f"OK  ({ds['n_timesteps']} timesteps, source={ds['source']})")
        return ds
    except Exception as exc:
        print(f"FAILED ({exc})")
        print("  [dataset] Falling back to SYNTHETIC_STANDIN (seed=42)...")
        ds = _make_synthetic_standin()
        print(
            f"  [dataset] SYNTHETIC_STANDIN ready: "
            f"{ds['n_timesteps']} timesteps, "
            f"{int(ds['labels'].sum())} anomalous points"
        )
        return ds
