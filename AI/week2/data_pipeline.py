#!/usr/bin/env python3
"""
data_pipeline.py  --  Week 2 (Person 1, AI): UK-DALE -> clean windows for NILMFormer.

It is a thin wrapper around the repo's own UKDALE_DataBuilder (src/helpers/preprocessing.py).
One run handles every appliance in the config, ONE APPLIANCE AT A TIME (one model per appliance).

For each appliance it
  1. checks which houses really have that appliance (exact name in labels.dat),
  2. builds non-overlapping windows with the repo builder (10 s resample, ffill 60 s, <5 W -> 0,
     clip at 6000 W, on/off status from the Kelly thresholds, mean to 1 min, windows with NaN dropped),
  3. splits train / validation / test (houses by the config; validation taken from the train windows),
  4. divides power by 6000 (fixed constant, NOT the data maximum),
  5. checks: no NaN, values in range, status 0/1, test houses only, window counts,
  6. saves float32 .npy files + settings.json to Drive, and plots a few windows.

Saved arrays have shape (N, 2, 2, L): axis1 = 0 aggregate, 1 appliance; axis2 = 0 power (/6000), 1 status (0/1).
The 8 timestamp channels are NOT stored: they are computed from the saved window start time
(Unix seconds, UTC) exactly as in timestamp_ref.py.

Usage (from the repo folder, e.g. /content/NILMFormer):
    uv run python data_pipeline.py --check_only                 # list houses/channels per appliance, no build
    uv run --with matplotlib python data_pipeline.py --preview  # build everything + model-input preview
Options: --window 64  --appliances kettle  --train_mode with_appliance  --valid_mode time  --force
Variant without house 5:  --appliances microwave --train_houses 1,3,4 --tag _no5
"""
import argparse
import datetime
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = ("train", "valid", "test")


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- config
def load_cfg(args):
    path = args.config or os.path.join(HERE, "nilm_config.json")
    with open(path) as f:
        cfg = json.load(f)
    if args.window:
        cfg["window_size"] = args.window
    if args.appliances:
        cfg["appliances"] = [a.strip() for a in args.appliances.split(",") if a.strip()]
    for k in ("data_dir", "out_dir", "train_mode", "valid_mode"):
        v = getattr(args, k)
        if v:
            cfg[k] = v
    if args.train_houses:
        cfg["train_houses"] = [int(x) for x in args.train_houses.split(",") if x.strip()]
    cfg["data_dir"] = os.path.join(cfg["data_dir"], "")      # builder concatenates strings: needs trailing slash
    assert cfg["train_mode"] in ("official", "with_appliance")
    assert cfg["valid_mode"] in ("random", "time")
    assert isinstance(cfg["window_size"], int) and cfg["window_size"] > 0
    assert not set(cfg["train_houses"]) & set(cfg["test_houses"]), "train and test houses overlap"
    return cfg


# --------------------------------------------------------------------------- houses / labels
def read_labels(data_dir, house):
    """labels.dat -> {name: channel_id} (first match wins, like the builder)."""
    p = os.path.join(data_dir, f"House{house}", "labels.dat")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{p} not found (expected folders House1..House5)")
    out = {}
    with open(p) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                out.setdefault(parts[1], int(parts[0]))
    return out


def house_channels(cfg, appliance, houses):
    """{house: channel id or None}. None = the builder would silently fill zeros."""
    res = {}
    for h in houses:
        labels = read_labels(cfg["data_dir"], h)
        ch = labels.get(appliance)
        if ch is not None and not os.path.isfile(os.path.join(cfg["data_dir"], f"House{h}", f"channel_{ch}.dat")):
            log(f"  !! House{h}: label '{appliance}' -> channel_{ch}.dat is listed but the FILE IS MISSING")
            ch = None
        res[h] = ch
    return res


def check_houses(cfg, appliance, verbose=True):
    allh = list(cfg["train_houses"]) + list(cfg["test_houses"])
    ch = house_channels(cfg, appliance, allh)
    if verbose:
        log(f"\n[{appliance}] houses and channels (exact name match in labels.dat)")
        for h in allh:
            role = "TEST " if h in cfg["test_houses"] else "train"
            labels = read_labels(cfg["data_dir"], h)
            near = [n for n in labels if appliance in n and n != appliance]
            if ch[h] is not None:
                log(f"  House{h} [{role}]  OK    channel_{ch[h]}.dat")
            else:
                hint = f"   (similar names, NOT used: {near})" if near else ""
                log(f"  House{h} [{role}]  NONE  -> target would be all zeros{hint}")
    return ch


# --------------------------------------------------------------------------- helpers
def to_unix(ts):
    """datetime column (naive = UTC) -> int64 Unix seconds, robust to pandas time units."""
    ts = pd.to_datetime(ts)
    return ((ts - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)


def build_windows(builder_cls, cfg, appliance, houses):
    """Repo builder -> (data float64 (N,2,2,L) in watts, start_unix (N,), house_id (N,))."""
    t0 = time.time()
    b = builder_cls(data_path=cfg["data_dir"], mask_app=appliance,
                    sampling_rate=cfg["sampling_rate"], window_size=cfg["window_size"])
    b.cutoff = cfg["power_divisor_w"]                         # clip at 6000 W (builder default is already 6000)
    data, st = b.get_nilm_dataset(house_indicies=list(houses))
    data = np.asarray(data)
    start = to_unix(st["start_date"])
    house = st.index.to_numpy().astype(np.int16)
    assert len(data) == len(start) == len(house)
    log(f"  built houses {list(houses)}: {len(data)} windows in {time.time() - t0:.0f}s")
    return data, start, house


def split_valid(cfg, house, start):
    """Indices (train_idx, valid_idx) into the train windows."""
    n = len(house)
    frac = cfg["valid_fraction"]
    if cfg["valid_mode"] == "random":
        # identical to the repo's split_train_test_nilmdataset(perc_house_test=frac, seed)
        np.random.seed(cfg["seed"])
        idx = np.arange(n)
        np.random.shuffle(idx)
        cut = int(n * (1 - frac))
        return idx[:cut], idx[cut:]
    tr, va = [], []                                           # "time": last `frac` of each house, in time order
    for h in np.unique(house):
        pos = np.where(house == h)[0]
        pos = pos[np.argsort(start[pos], kind="stable")]
        cut = int(len(pos) * (1 - frac))
        tr.append(pos[:cut])
        va.append(pos[cut:])
    return np.concatenate(tr), np.concatenate(va)


def check_array(name, data_w, start, house, cfg, allowed_houses):
    """Hard checks on raw-watt data (before scaling). Raises AssertionError on any problem."""
    L, div = cfg["window_size"], cfg["power_divisor_w"]
    assert data_w.ndim == 4 and data_w.shape[1:] == (2, 2, L), f"{name}: bad shape {data_w.shape}"
    assert not np.isnan(data_w).any(), f"{name}: NaN found"
    for c, label in ((0, "aggregate"), (1, "appliance")):
        p = data_w[:, c, 0, :]
        assert p.min() >= 0 and p.max() <= div + 1e-6, f"{name}: {label} power outside [0,{div}] W ({p.min()}, {p.max()})"
        assert set(np.unique(data_w[:, c, 1, :])) <= {0.0, 1.0}, f"{name}: {label} status not 0/1"
    assert set(np.unique(house)) <= set(allowed_houses), f"{name}: unexpected houses {np.unique(house)}"
    assert (start % 60 == 0).all(), f"{name}: a start time is not on a whole minute"


def summarize(data_w, start, house, split):
    rows = []
    for h in np.unique(house):
        m = house == h
        app = data_w[m, 1, 0, :]
        on = (data_w[m, 1, 1, :] > 0).any(axis=1)
        rows.append(dict(split=split, house=int(h), windows=int(m.sum()), active_windows=int(on.sum()),
                         pct_active=round(100.0 * on.mean(), 1), app_mean_W=round(float(app.mean()), 1),
                         app_max_W=round(float(app.max()), 0),
                         first_start_utc=str(pd.to_datetime(int(start[m].min()), unit="s")),
                         last_start_utc=str(pd.to_datetime(int(start[m].max()), unit="s"))))
    return rows


def save_split(folder, split, data_scaled, start, house):
    np.save(os.path.join(folder, f"{split}_data.npy"), data_scaled)
    np.save(os.path.join(folder, f"{split}_start.npy"), start)
    np.save(os.path.join(folder, f"{split}_house.npy"), house)


def plot_windows(folder, appliance, data_w, start, cfg, n_active=3):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  (matplotlib not installed: skipping plots; use  uv run --with matplotlib python ...)")
        return
    on = np.where((data_w[:, 1, 1, :] > 0).any(axis=1))[0]
    rng = np.random.default_rng(cfg["seed"])
    pick = list(rng.choice(on, size=min(n_active, len(on)), replace=False)) if len(on) else []
    pick.append(int(rng.integers(0, len(data_w))))
    fig, axes = plt.subplots(2, len(pick), figsize=(4.2 * len(pick), 5), sharex=True, squeeze=False)
    x = np.arange(data_w.shape[-1])
    for j, i in enumerate(pick):
        axes[0, j].plot(x, data_w[i, 0, 0], lw=1)
        axes[0, j].set_title(f"test window {i}\nstart {pd.to_datetime(int(start[i]), unit='s')} UTC", fontsize=8)
        axes[1, j].plot(x, data_w[i, 1, 0], lw=1, color="C1")
        axes[1, j].fill_between(x, 0, data_w[i, 1, 1] * max(1.0, data_w[i, 1, 0].max()), alpha=0.2, color="C2", step="mid")
        axes[1, j].set_xlabel("minute in window")
    axes[0, 0].set_ylabel("aggregate (W)")
    axes[1, 0].set_ylabel(f"{appliance} (W), green = on")
    fig.tight_layout()
    path = os.path.join(folder, f"alignment_{appliance}_w{cfg['window_size']}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log(f"  plot saved: {path}")


def preview(cfg, folder, appliance, data_scaled, start):
    """Run real test windows through the repo's NILMDataset and compare with timestamp_ref (the C spec)."""
    try:
        from src.helpers.dataset import NILMDataset
    except Exception as e:                                    # e.g. torch not available
        log(f"  (preview skipped: cannot import NILMDataset: {e})")
        return
    sys.path.insert(0, HERE)
    from timestamp_ref import exo_channels
    st = pd.DataFrame({"start_date": pd.to_datetime(start, unit="s")})
    ds = NILMDataset(data_scaled, st_date=st, list_exo_variables=cfg["exo_variables"], freq=cfg["sampling_rate"])
    on = np.where((data_scaled[:, 1, 1, :] > 0).any(axis=1))[0]
    other = np.linspace(0, len(ds) - 1, 5).astype(int)
    idx = list(dict.fromkeys(list(on[: 3]) + list(other)))[:5]
    X, Y, S, worst = [], [], [], 0.0
    for i in idx:
        x, y, _ = ds[i]
        x = np.asarray(x, dtype=np.float32)
        assert x.shape == (9, cfg["window_size"]), x.shape
        worst = max(worst, float(np.abs(x[1:] - exo_channels(start[i], cfg["window_size"])).max()))
        X.append(x); Y.append(np.asarray(y, dtype=np.float32)[0]); S.append(start[i])
    assert worst < 1e-6, f"repo timestamp channels differ from timestamp_ref (max diff {worst})"
    np.save(os.path.join(folder, "testvec_input.npy"), np.stack(X))
    np.save(os.path.join(folder, "testvec_target.npy"), np.stack(Y))
    np.save(os.path.join(folder, "testvec_start.npy"), np.array(S, dtype=np.int64))
    log(f"  preview OK: model input shape {X[0].shape}; timestamp channels match timestamp_ref "
        f"(max diff {worst:.1e}); {len(X)} test vectors saved")


# --------------------------------------------------------------------------- one appliance
def run_appliance(cfg, appliance, args, builder_cls):
    L = cfg["window_size"]
    folder = os.path.join(cfg["out_dir"], f"w{L}", appliance + args.tag)
    if os.path.isfile(os.path.join(folder, "settings.json")) and not args.force:
        log(f"\n[{appliance}] already built in {folder} (use --force to rebuild)")
        return
    os.makedirs(folder, exist_ok=True)
    log(f"\n==================== {appliance}  (window {L}) ====================")

    chans = check_houses(cfg, appliance)
    for h in cfg["test_houses"]:
        if chans[h] is None and not args.allow_missing_test:
            raise SystemExit(f"STOP: test house {h} has no exact '{appliance}' channel, so the test target would be all zeros.")
    train_houses = [h for h in cfg["train_houses"] if cfg["train_mode"] == "official" or chans[h] is not None]
    zero_houses = [h for h in train_houses if chans[h] is None]
    if zero_houses:
        log(f"  WARNING: train houses {zero_houses} have no '{appliance}': their windows get all-zero targets "
            f"(this is what the official script does; use --train_mode with_appliance to exclude them)")
    if not train_houses:
        raise SystemExit(f"STOP: no training house for {appliance}")

    log(f"building train houses {train_houses} and test houses {cfg['test_houses']} ...")
    tr_w, tr_start, tr_house = build_windows(builder_cls, cfg, appliance, train_houses)
    te_w, te_start, te_house = build_windows(builder_cls, cfg, appliance, cfg["test_houses"])

    tr_idx, va_idx = split_valid(cfg, tr_house, tr_start)
    parts = {
        "train": (tr_w[tr_idx], tr_start[tr_idx], tr_house[tr_idx]),
        "valid": (tr_w[va_idx], tr_start[va_idx], tr_house[va_idx]),
        "test": (te_w, te_start, te_house),
    }
    del tr_w
    assert not (set(map(int, np.unique(parts["test"][2]))) & set(train_houses)), "test house also in training"

    rows = []
    for split in SPLITS:
        w, s, h = parts[split]
        allowed = cfg["test_houses"] if split == "test" else train_houses
        check_array(f"{appliance}/{split}", w, s, h, cfg, allowed)
        rows += summarize(w, s, h, split)
    table = pd.DataFrame(rows)
    log("\nwindow counts and activity (watts, before scaling):")
    log(table.drop(columns=["first_start_utc", "last_start_utc"]).to_string(index=False))
    log(f"totals: " + ", ".join(f"{sp}={len(parts[sp][0])}" for sp in SPLITS))
    for sp in SPLITS:
        assert len(parts[sp][0]) > 0, f"{appliance}/{sp} is empty"
    if int((parts["test"][0][:, 1, 1, :] > 0).any(axis=1).sum()) == 0:
        log("  WARNING: no active test window for this appliance")

    div = np.float32(cfg["power_divisor_w"])
    for split in SPLITS:
        w, s, h = parts[split]
        scaled = w.astype(np.float32)
        scaled[:, :, 0, :] /= div                             # power channels only; status stays 0/1
        assert scaled[:, :, 0, :].min() >= 0 and scaled[:, :, 0, :].max() <= 1.0 + 1e-6
        assert not np.isnan(scaled).any()
        save_split(folder, split, scaled, s, h)
        parts[split] = (w, s, h, scaled)
    log(f"\nsaved to {folder}")

    settings = {
        "appliance": appliance, "tag": args.tag, "dataset": cfg["dataset"], "window_size": L,
        "sampling_rate": cfg["sampling_rate"], "power_divisor_w": cfg["power_divisor_w"],
        "exo_variables": cfg["exo_variables"], "array_layout": "(N, 2[aggregate, appliance], 2[power/6000, status], L)",
        "start_time": "int64 Unix seconds, UTC, one per window (timestamp channels = timestamp_ref.py)",
        "train_houses_used": train_houses, "train_mode": cfg["train_mode"], "test_houses": cfg["test_houses"],
        "valid_mode": cfg["valid_mode"], "valid_fraction": cfg["valid_fraction"], "seed": cfg["seed"],
        "houses_channel": {str(h): c for h, c in chans.items()},
        "builder": "src/helpers/preprocessing.py UKDALE_DataBuilder (10 s resample, ffill limit 6 = 60 s, <5 W -> 0, "
                   "clip 6000 W, Kelly on/off thresholds, mean to sampling_rate, non-overlapping windows, windows with NaN dropped)",
        "counts": {sp: int(len(parts[sp][0])) for sp in SPLITS},
        "per_house": rows, "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "command": " ".join(sys.argv),
    }
    with open(os.path.join(folder, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)

    if not args.no_plots:
        plot_windows(folder, appliance, parts["test"][0], parts["test"][1], cfg)
    if args.preview:
        preview(cfg, folder, appliance, parts["test"][3], parts["test"][1])


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config"); ap.add_argument("--repo", default=".", help="NILMFormer repo folder (contains src/)")
    ap.add_argument("--window", type=int); ap.add_argument("--appliances", help="comma list, e.g. kettle,fridge")
    ap.add_argument("--data_dir"); ap.add_argument("--out_dir")
    ap.add_argument("--train_houses", help="override the config, e.g. 1,3,4 (use with --tag so the official run is not overwritten)")
    ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. _no5 -> processed/w128/microwave_no5")
    ap.add_argument("--train_mode", choices=["official", "with_appliance"])
    ap.add_argument("--valid_mode", choices=["random", "time"])
    ap.add_argument("--check_only", action="store_true", help="only list houses/channels per appliance")
    ap.add_argument("--preview", action="store_true", help="also run windows through NILMDataset and save test vectors")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--allow_missing_test", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args)
    log(f"config: window={cfg['window_size']} rate={cfg['sampling_rate']} divisor={cfg['power_divisor_w']} "
        f"appliances={cfg['appliances']} train={cfg['train_houses']} test={cfg['test_houses']} "
        f"train_mode={cfg['train_mode']} valid_mode={cfg['valid_mode']}")

    if args.check_only:
        for a in cfg["appliances"]:
            check_houses(cfg, a)
        return

    sys.path.insert(0, os.path.abspath(args.repo))
    from src.helpers.preprocessing import UKDALE_DataBuilder
    for a in cfg["appliances"]:
        run_appliance(cfg, a, args, UKDALE_DataBuilder)
    log("\nDONE.")


if __name__ == "__main__":
    main()
