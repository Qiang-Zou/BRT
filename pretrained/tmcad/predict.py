"""Predict a class for one compatible preprocessed face/topology cache pair."""
from pathlib import Path
import argparse
import json
import os
import random
import tempfile

import numpy as np
import torch
from load_model import HERE, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--face', type=Path, required=True, help='Historical triangular face .bin cache')
    parser.add_argument('--topo', type=Path, required=True, help='Matching historical topology .bin cache')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    face, topo = args.face.resolve(), args.topo.resolve()
    for path in [face, topo]:
        if not path.is_file(): raise FileNotFoundError(path)
    output = args.output.resolve() if args.output else None
    if output and output.exists(): raise FileExistsError(output)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    with tempfile.TemporaryDirectory(prefix='brt_predict_') as folder:
        folder = Path(folder)
        (folder / 'logs').mkdir()
        (folder / 'datasplit.json').write_text(json.dumps({'test': [dict(face=str(face), topo=str(topo), label=0)]}))
        previous = Path.cwd()
        try:
            os.chdir(folder)
            from datasets.brt_dataset import BRTDataset_cls_online
            model = load_model(checkpoint, args.device)
            random.seed(0); np.random.seed(0); torch.manual_seed(0)
            ds = BRTDataset_cls_online(root_dir=str(folder), split='test', masking_rate=None, masking_rate_v2=None)
            if len(ds) != 1: raise ValueError('Cache does not satisfy the historical loader limits')
            batch = ds._collate([ds[0]])
            inputs = {k: v if k == 'edge_index_length' else v.to(args.device)
                      for k, v in batch.items() if k not in ['filename', 'label', 'points']}
            with torch.no_grad(): logits = model(inputs)
            assert logits.shape == (1, 10) and torch.isfinite(logits).all()
            probabilities = torch.softmax(logits[0], dim=-1).cpu().tolist()
        finally:
            os.chdir(previous)
    classes = json.loads((HERE / 'data/class_names.json').read_text())
    pred = int(np.argmax(probabilities))
    result = dict(class_id=pred, class_name=classes[str(pred)],
                  probabilities={classes[str(i)]: value for i, value in enumerate(probabilities)},
                  device=args.device, input_type='historical_processed_cache_pair')
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
