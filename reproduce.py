"""Offline GraphML CAN IDS reproduction using the public ROAD ZIP."""

import argparse
import hashlib
import json
import platform
import re
import time
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from node2vec import Node2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, average_precision_score, confusion_matrix, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


SEED = 42
ROAD_MD5 = "cab184cfc2fe12c0834bc46188c0f330"
MAX_ID = 0x700
CHUNK = 160_000
SCENARIOS = {
    "correlated_signal": 3, "max_engine_coolant_temp": 1, "max_speedometer": 3,
    "reverse_light_off": 3, "reverse_light_on": 3,
}
EMBEDDING = [f"embedding_{i}" for i in range(64)]
SIGNAL = [f"signal_{i}_{stat}" for i in range(1, 23) for stat in ("mean", "std")]
PACKET = re.compile(rb"^\s*\(([\d.]+)\)\s+\S+\s+([\da-fA-F]+)#", re.MULTILINE)


def captures(road, ambient_name=None):
    with road.open("road/attacks/capture_metadata.json") as fh:
        metadata = json.load(fh)
    ambient = sorted(name for name in road.namelist()
                     if name.startswith("road/ambient/") and name.endswith(".log"))
    if len(ambient) != 12:
        raise ValueError(f"Expected 12 ambient captures, found {len(ambient)}")
    if ambient_name:
        ambient = [p for p in ambient if Path(p).stem == ambient_name]
        if not ambient:
            raise ValueError(f"Unknown ambient capture: {ambient_name}")
    for path in ambient:
        yield "ambient", path, None
    for kind, count in SCENARIOS.items():
        for index in range(1, count + 1):
            name = f"{kind}_attack{f'_{index}' if count > 1 else ''}_masquerade"
            yield kind, f"road/attacks/{name}.log", metadata[name]["injection_interval"]


def read_log(road, path):
    """Read timestamp and CAN ID only; retain capture order, including ties."""
    times, ids = [], []
    with road.open(path) as fh:
        while block := fh.read(4 * 2**20):
            block += fh.readline()
            rows = PACKET.findall(block)
            if len(rows) != len(block.splitlines()):
                raise ValueError(f"Unexpected candump row in {path}")
            times.append(np.fromiter((float(t) for t, _ in rows), np.float64, len(rows)))
            ids.append(np.fromiter((int(v, 16) for _, v in rows), np.uint32, len(rows)))
    ts, pid = np.concatenate(times), np.concatenate(ids)
    if ts.size < 2 or np.any(ts[1:] < ts[:-1]):
        raise ValueError(f"Expected chronological timestamps in {path}")
    mask = pid <= MAX_ID
    return (ts[mask] - ts[0]), pid[mask]


def windows(ts, size, offset, unit):
    if unit == "frames":
        left = np.arange(0, len(ts) - int(size) + 1, int(offset))
        right = left + int(size)
        return left, right, ts[left], np.nextafter(ts[right - 1], np.inf)
    start = np.arange(0, max(0, ts[-1] - size) + offset, offset)
    start = start[start + size <= ts[-1]]
    end = start + size
    return np.searchsorted(ts, start), np.searchsorted(ts, end), start, end


def message_graph(ids):
    graph = nx.DiGraph()
    graph.add_nodes_from(dict.fromkeys(map(int, ids)))
    if ids.size > 1:
        u, counts = np.unique(ids[:-1] * (MAX_ID + 1) + ids[1:], return_counts=True)
        graph.add_weighted_edges_from(
            (*divmod(int(code), MAX_ID + 1), int(n)) for code, n in zip(u, counts)
        )
    return graph


class SerialSkipGram(Word2Vec):
    """Run Gensim 4.4's single worker inline, preserving its exact batch order."""

    def _train_epoch(self, data_iterable, cur_epoch=0, total_examples=None,
                     total_words=None, **kwargs):
        tally = [0, 0, 0]
        memory = self._get_thread_working_mem()
        started = time.perf_counter()

        class InlineQueue:
            def put(queue, item):
                if item is not None:
                    sentences, alpha = item
                    effective, raw = self._do_train_job(sentences, alpha, memory)
                    tally[0] += effective
                    tally[1] += raw
                    tally[2] += 1

        self._job_producer(data_iterable, InlineQueue(), cur_epoch, total_examples, total_words)
        self.total_train_time += time.perf_counter() - started
        return tuple(tally)


def graph_embedding(ids):
    graph = message_graph(ids)
    n2v = Node2Vec(graph, dimensions=64, walk_length=15, num_walks=100,
                   p=1.5, q=0.5, workers=1, seed=SEED, quiet=True)
    model = SerialSkipGram(n2v.walks, vector_size=n2v.dimensions, sg=1, window=10,
                           min_count=1, batch_words=7, seed=SEED, workers=1)
    return model.wv[[str(node) for node in graph]].mean(axis=0)


def signal_vector(frame, ids, cols):
    frame = frame[frame["ID"].isin(ids)]
    if frame.empty:
        return np.zeros(len(SIGNAL))
    grouped = frame.groupby("ID", sort=False)
    g = grouped[cols]
    mean, std = g.mean(), g.std()
    counts = g.count()
    gaps = (counts.gt(0) & counts.lt(grouped.size(), axis=0)).any(axis=1)
    for aid in gaps.index[gaps]:
        values = grouped.get_group(aid)[cols].interpolate(limit_area="inside")
        mean.loc[aid], std.loc[aid] = values.mean(), values.std()
    pair = np.column_stack([mean.round(2).sum(), std.round(2).sum()]) / len(ids)
    return pair.ravel()


def read_signal_windows(road, path, start, end, ids, left, right):
    """Evaluate signal statistics inside the same capture-local time bounds."""
    cols = [f"Signal_{i}_of_ID" for i in range(1, 23)]
    with road.open(path) as fh:
        chunks = iter(pd.read_csv(fh, usecols=["Time", "ID", *cols], chunksize=CHUNK))
        block = next(chunks)
        stamps = block["Time"].to_numpy()
        for a, b, l, r in zip(start, end, left, right):
            while b > stamps[-1]:
                new = next(chunks, None)
                if new is None:
                    break
                if new["Time"].iat[0] < stamps[-1]:
                    raise ValueError(f"Unexpected timestamp order in {path}")
                block = pd.concat([block.iloc[stamps.searchsorted(a):], new], ignore_index=True)
                stamps = block["Time"].to_numpy()
            i, j = stamps.searchsorted([a, b])
            yield signal_vector(block.iloc[i:j], np.unique(ids[l:r]), cols)


def extract(args, protocol):
    cache = args.out / "capture-features"
    cache.mkdir(exist_ok=True)
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    frames = []
    with ZipFile(args.road) as road, Parallel(n_jobs=args.jobs) as pool:
        for kind, path, interval in captures(road, args.ambient):
            start_clock = time.perf_counter()
            name = Path(path).stem
            cached = cache / f"{key}-{name}.csv"
            if cached.exists():
                frame = pd.read_csv(cached, float_precision="round_trip")
                frames.append(frame)
                print(f"{name}: reused {len(frame)} windows", flush=True)
                continue
            ts, ids = read_log(road, path)
            left, right, start, end = windows(ts, args.window, args.offset, args.unit)
            valid = right - left >= 2
            left, right, start, end = (a[valid] for a in (left, right, start, end))
            if not left.size:
                raise ValueError(f"No usable full windows in {path}; reduce window size")
            branch = "ambient" if interval is None else "attacks"
            csv = f"road/signal_extractions/{branch}/{name}.csv"
            sig = list(read_signal_windows(road, csv, start, end, ids, left, right))
            emb = []
            for i in range(0, len(left), 512):
                j = i + 512
                emb.extend(pool(delayed(graph_embedding)(ids[a:b])
                                for a, b in zip(left[i:j], right[i:j])))
                if len(left) > 512:
                    print(f"{name}: embedded {len(emb)}/{len(left)} windows", flush=True)
            features = pd.DataFrame(np.column_stack([emb, sig]), columns=EMBEDDING + SIGNAL)
            labels = np.zeros(len(start), dtype=int)
            if interval is not None:
                labels[(start <= interval[1]) & (end > interval[0])] = 1
            features.insert(0, "label", labels)
            features.insert(0, "end_sec", end)
            features.insert(0, "start_sec", start)
            features.insert(0, "scenario", kind)
            features.insert(0, "capture", name)
            tmp = cached.with_suffix(".part")
            features.to_csv(tmp, index=False)
            tmp.replace(cached)
            frames.append(features)
            print(f"{name}: {len(start)} windows, {labels.sum()} positive, "
                  f"{time.perf_counter() - start_clock:.1f}s", flush=True)
    return pd.concat(frames, ignore_index=True)


def graph_timing(args):
    rows = []
    with ZipFile(args.road) as road:
        for kind, path, _ in captures(road, args.ambient):
            ts, ids = read_log(road, path)
            left, right, start, end = windows(ts, args.window, args.offset, args.unit)
            for a, b, s, e in zip(left, right, start, end):
                if b - a < 2:
                    continue
                t = time.perf_counter_ns()
                graph = message_graph(ids[a:b])
                ms = (time.perf_counter_ns() - t) / 1e6
                rows.append((Path(path).stem, kind, s, e, b - a,
                             graph.number_of_nodes(), graph.number_of_edges(), ms))
    data = pd.DataFrame(rows, columns=["capture", "scenario", "start_sec", "end_sec",
                                       "frames", "nodes", "edges", "graph_ms"])
    data.to_csv(args.out / "graph-timing.csv", index=False, float_format="%.8f")
    summary = data.groupby("scenario").graph_ms.agg(
        windows="count", mean_ms="mean", median_ms="median",
        p95_ms=lambda x: x.quantile(0.95))
    summary.to_csv(args.out / "graph-timing-summary.csv", float_format="%.8f")
    print(summary.to_string(), flush=True)


def metrics(y, scores):
    pred = scores >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, scores)
    return dict(windows=len(y), positives=int(y.sum()),
                roc_auc=roc_auc_score(y, scores),
                auc_pr=auc(recall, precision),
                average_precision=average_precision_score(y, scores),
                accuracy=(tn + tp) / len(y), precision=tp / max(1, tp + fp),
                recall=tp / (tp + fn), f1=2 * tp / (2 * tp + fp + fn),
                mcc=matthews_corrcoef(y, pred),
                fpr=fp / (fp + tn), fnr=fn / (fn + tp),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def evaluate(data, args):
    y = data.label.to_numpy()
    preds = data[["capture", "scenario", "start_sec", "end_sec", "label"]].copy()
    variants = {
        "graph": EMBEDDING, "signals": SIGNAL, "hybrid": EMBEDDING + SIGNAL,
    }
    classifiers = {"rf": lambda: RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_split=6, min_samples_leaf=2,
        random_state=SEED, n_jobs=args.jobs)}
    if "xgb" in args.models:
        from xgboost import XGBClassifier
        classifiers["xgb"] = lambda: XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric="logloss",
            random_state=SEED, n_jobs=args.jobs)
    rows = []
    for protocol in args.cv:
        if protocol == "capture":
            splitter = StratifiedGroupKFold(n_splits=5)
        else:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        folds = list(splitter.split(data, y, groups=data.capture if protocol == "capture" else None))
        for fold, (train, test) in enumerate(folds):
            if protocol == "capture":
                assert set(data.capture.iloc[train]).isdisjoint(data.capture.iloc[test])
            if np.unique(y[train]).size != 2:
                raise ValueError(f"Training fold {fold} of {protocol} has only one class")
            preds.loc[test, f"{protocol}_fold"] = fold
        for variant, cols in variants.items():
            x = data[cols].to_numpy(dtype=np.float32)
            resampled = []
            for train, test in folds:
                neighbors = min(5, int(np.bincount(y[train]).min()) - 1)
                if neighbors < 1:
                    raise ValueError("Not enough minority training examples for SMOTE")
                scaler = MinMaxScaler().fit(x[train])
                x_fit, y_fit = SMOTE(random_state=SEED, k_neighbors=neighbors).fit_resample(
                    scaler.transform(x[train]), y[train])
                resampled.append((x_fit, y_fit, scaler.transform(x[test]), test))
            for model_name in args.models:
                scores = np.full(len(y), np.nan)
                for x_fit, y_fit, x_test, test in resampled:
                    model = classifiers[model_name]().fit(x_fit, y_fit)
                    scores[test] = model.predict_proba(x_test)[:, 1]
                assert np.all(np.isfinite(scores))
                preds[f"{protocol}_{model_name}_{variant}"] = scores
                for kind in ["pooled", *SCENARIOS]:
                    mask = np.ones(len(y), dtype=bool) if kind == "pooled" else (
                        data.scenario == kind).to_numpy()
                    if np.unique(y[mask]).size == 2:
                        rows.append(dict(cv=protocol, model=model_name, features=variant,
                                         scenario=kind, **metrics(y[mask], scores[mask])))
    return pd.DataFrame(rows), preds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--road", type=Path, required=True, help="Verified Zenodo road.zip")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reuse-features", action="store_true",
                   help="Re-evaluate existing features.csv with identical window settings")
    p.add_argument("--ttw-only", action="store_true",
                   help="Measure graph construction only; skip embeddings and classification")
    p.add_argument("--unit", choices=["seconds", "frames"], default="seconds")
    p.add_argument("--ambient", help="Normal capture stem; default: all 12 recordings")
    p.add_argument("--window", type=float, default=8)
    p.add_argument("--offset", type=float, default=8)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--models", nargs="+", choices=["rf", "xgb"], default=["rf", "xgb"])
    p.add_argument("--cv", nargs="+", choices=["capture", "window"], default=["capture", "window"])
    args = p.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.window, args.offset)) or args.jobs < 1:
        p.error("window, offset and jobs must be positive")
    if args.unit == "frames" and (args.window < 2 or not all(
            v.is_integer() if isinstance(v, float) else True for v in (args.window, args.offset))):
        p.error("frames requires integer window and offset, with window >= 2")
    args.out.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()
    features_path = args.out / "features.csv"
    protocol_path = args.out / "features-config.json"
    with args.road.open("rb") as road:
        md5 = hashlib.file_digest(road, "md5").hexdigest()
    if md5 != ROAD_MD5:
        p.error("ROAD checksum mismatch; use road.zip from Zenodo record 10462796")
    if args.ttw_only:
        if args.reuse_features:
            p.error("--ttw-only and --reuse-features cannot be combined")
        graph_timing(args)
        return
    protocol = {k: getattr(args, k) for k in ("unit", "window", "offset")} | {
        "road_md5": md5, "seed": SEED}
    if args.ambient:
        protocol["ambient"] = args.ambient
    if args.reuse_features:
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous.get("ambient") != args.ambient or any(
                previous.get(k) != v for k, v in protocol.items()):
            p.error("--reuse-features requires the original dataset and window settings")
        data = pd.read_csv(features_path, float_precision="round_trip")
        seconds = previous["feature_wall_seconds"]
    else:
        data = extract(args, protocol)
        seconds = round(time.perf_counter() - clock, 2)
        data.to_csv(features_path, index=False)
        protocol_path.write_text(json.dumps(protocol | {"feature_wall_seconds": seconds},
                                           indent=2) + "\n", encoding="utf-8")
    scores, preds = evaluate(data, args)
    scores.to_csv(args.out / "metrics.csv", index=False, float_format="%.8f")
    preds.to_csv(args.out / "predictions.csv", index=False)
    config = vars(args) | {"road": str(args.road), "out": str(args.out),
                           "seed": SEED, "road_md5": md5, "feature_wall_seconds": seconds,
                           "captures": data.capture.nunique(), "windows": len(data),
                           "python": platform.python_version(),
                           "packages": {p: version(p) for p in ("node2vec", "gensim", "numpy",
                           "networkx", "pandas", "scikit-learn", "imbalanced-learn", "xgboost")}}
    (args.out / "run.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(scores[scores.scenario == "pooled"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
