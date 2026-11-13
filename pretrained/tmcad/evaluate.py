"""Portable wrapper for the recorded CPU evaluation. Cache files are required.

Default settings preserve the archived test evaluation: seed 0, eight CPU threads,
eight loader workers, batch 16, highest FP32 precision, and no dropped test tail.
This wrapper was added on 2026-11-13; it is not the historical training entry point.
"""
from pathlib import Path
import argparse
import json
import os
import random
import sys
import time
import hashlib

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from load_model import load_model


def seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-root", type=Path, required=True)
    parser.add_argument("--topo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Released tensor-only checkpoint")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--smoke", action="store_true", help="One validation batch; never a test metric")
    args = parser.parse_args()
    if args.smoke and args.split != "val":
        parser.error("--smoke requires --split val")
    out = args.output_dir.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if (out / "result.json").exists():
        parser.error("Use a new output directory; prior results are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    raw = json.loads((HERE / "data/datasplit_relative.json").read_text())
    roots = {"face": args.face_root.resolve(), "topo": args.topo_root.resolve()}
    mapped = {split: [dict(item, **{k: str(roots[k] / item[k]) for k in roots})
                      for item in items] for split, items in raw.items()}
    for item in mapped[args.split]:
        for key in roots:
            if not Path(item[key]).is_file():
                raise FileNotFoundError(f"Missing {key} cache: {item[key]}")
    (out / "datasplit.json").write_text(json.dumps(mapped))
    os.chdir(out)  # The unchanged upstream loader writes logs/dataset.log.
    from datasets.brt_dataset import BRTDataset_cls_online

    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    seed()
    model = load_model(checkpoint)
    ds = BRTDataset_cls_online(root_dir=str(out), split=args.split,
                              masking_rate=None, masking_rate_v2=None)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8,
                        collate_fn=ds._collate, drop_last=False,
                        generator=torch.Generator().manual_seed(0))
    seed()
    if args.split == "test":
        if len(ds) != 794:
            raise RuntimeError(f"Expected 794 admitted test CADs, got {len(ds)}")
        excluded = set(json.loads((HERE / "data/excluded_test_face_paths.json").read_text()))
        expected = [x for x in raw["test"] if x["face"] not in excluded]
        observed = [(str(Path(x["face"]).relative_to(roots["face"])), x["label"]) for x in ds.data]
        if observed != [(x["face"], x["label"]) for x in expected]:
            raise RuntimeError("Admitted test membership/order differs from the archived evaluation")
    cm = torch.zeros(10, 10, dtype=torch.int64)
    total = 0
    total_loss = 0.0
    predictions = []
    started = time.time()
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].long()
            inputs = {k: v for k, v in batch.items() if k not in ["filename", "label", "points"]}
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("Non-finite model output")
            pred = logits.argmax(-1)
            cm += torch.bincount(labels * 10 + pred, minlength=100).reshape(10, 10)
            total += len(labels)
            total_loss += float(loss) * len(labels)
            predictions.extend(dict(face=str(Path(name).relative_to(roots['face'])), label=int(a), prediction=int(b))
                               for name, a, b in zip(batch['filename'], labels, pred))
            if args.smoke:
                break
    if args.smoke:
        result = dict(status="validation_forward_smoke_passed", processed=total,
                      finite_logits=True, research_metric=False)
    else:
        assert total == len(ds)
        result = dict(status="completed", split=args.split, num_samples=total,
                      correct=int(cm.diag().sum()), accuracy=100 * int(cm.diag().sum()) / total,
                      loss=total_loss / total, confusion_matrix=cm.tolist(),
                      seconds=time.time() - started, strict_paper_reproduction=False,
                      checkpoint_sha256=json.loads((HERE / 'model_card.json').read_text())['state_dict_sha256'],
                      relative_manifest_sha256=hashlib.sha256((HERE / 'data/datasplit_relative.json').read_bytes()).hexdigest(),
                      device='cpu', batch_size=16, workers=8, cpu_threads=8, seed=0, drop_last=False)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
