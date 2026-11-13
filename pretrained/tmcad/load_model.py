"""Load the published tensor-only TMCAD fine-tuned checkpoint."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'runtime'))
import torch
from models.brt_classfication import BRTClassification


def load_model(checkpoint=None, device='cpu'):
    metadata = json.loads((HERE / 'model_card.json').read_text())
    checkpoint = Path(checkpoint) if checkpoint else HERE / metadata['checkpoint']
    if not checkpoint.is_file():
        raise FileNotFoundError('Restore the bundled checkpoint with download_weights.py or supply a checkpoint at ' + str(checkpoint))
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata['state_dict_sha256']:
        raise ValueError('Checkpoint SHA-256 differs from the released 83.88% model')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = BRTClassification(num_classes=10, num_control_pts=28)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite checkpoint tensor')
    return model.to(device).eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    print(json.dumps(dict(strict_load=True, parameters=sum(p.numel() for p in model.parameters()),
                          finite=True, device=args.device, inference_run=False)))
