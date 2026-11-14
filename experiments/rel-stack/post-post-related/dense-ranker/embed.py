"""Embed every rel-stack question's initial title + body with a frozen sentence encoder.

Writes ``<out>/<slug>.npz`` (``ids``, ``times``, ``emb`` float16) for
:class:`relconduit.post_embed.DenseIndex`. Texts are :func:`relconduit.post_embed.initial_post_texts`
over ``postHistory`` rows up to the test timestamp (``task.get_db()``'s cut); each vector's
time is the latest revision row it came from, and the index only ever searches vectors dated
strictly before a seed time. Resumes from ``<out>/<slug>.part<k>.npy`` chunks.

    uv run python experiments/rel-stack/post-post-related/dense-ranker/embed.py \\
        --model sentence-transformers/all-MiniLM-L6-v2 --out runs/p29/emb
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import relbench

from relconduit.post_embed import embed_texts, initial_post_texts


def slug(model: str) -> str:
    return model.split("/")[-1]


def question_texts(limit: int | None = None) -> pd.DataFrame:
    ds = relbench.load_dataset("stanford-star/relbench-v1/rel-stack")
    task = ds.load_task("post-post-related")
    cut = pd.Timestamp(ds.test_timestamp)
    db = task.get_db()  # rows up to the test timestamp, as the ranker sees them
    posts = db.table_dict["posts"].df
    posts = posts[posts["PostTypeId"] == 1][["Id", "PostTypeId", "ParentId"]]
    ph = db.table_dict["postHistory"].df
    ph = ph[ph["PostHistoryTypeId"].isin([1, 2]) & ph["PostId"].isin(posts["Id"])]
    ph = ph[["Id", "PostId", "PostHistoryTypeId", "Text", "CreationDate"]]
    assert ph["CreationDate"].max() <= cut
    tx = initial_post_texts(ph, posts)
    return tx.head(limit) if limit else tx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="runs/p29/emb")
    p.add_argument("--chunk", type=int, default=20_000)
    p.add_argument("--limit", type=int, help="smoke: only this many posts")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-seq-length", type=int, default=256)
    a = p.parse_args()
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = slug(a.model) + (f"_smoke{a.limit}" if a.limit else "")
    final = out / f"{name}.npz"
    if final.exists():
        print(f"{final} exists", flush=True)
        return
    t0 = time.time()
    tx = question_texts(a.limit)
    print(f"[{time.time() - t0:.0f}s] {len(tx):,} question texts", flush=True)
    parts = []
    for k, start in enumerate(range(0, len(tx), a.chunk)):
        path = out / f"{name}.part{k}.npy"
        if not path.exists():
            t1 = time.time()
            texts = tx["text"].iloc[start : start + a.chunk].tolist()
            emb = embed_texts(
                texts,
                a.model,
                batch_size=a.batch_size,
                max_seq_length=a.max_seq_length,
            ).astype(np.float16)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, emb)
            tmp.rename(path)
            rate = len(texts) / (time.time() - t1)
            print(
                f"[{time.time() - t0:.0f}s] {name} chunk {k}: {len(texts):,} texts, {rate:.0f}/s",
                flush=True,
            )
        parts.append(np.load(path))
    emb = np.concatenate(parts)
    tmp = out / f"{name}.tmp.npz"
    np.savez(tmp, ids=tx["Id"].to_numpy(), times=tx["t"].to_numpy(), emb=emb)
    tmp.rename(final)
    for k in range(len(parts)):
        (out / f"{name}.part{k}.npy").unlink()
    print(f"[{time.time() - t0:.0f}s] wrote {final} {emb.shape}", flush=True)


if __name__ == "__main__":
    main()
