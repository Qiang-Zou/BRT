# BRT Models

| Dataset | Task | ACC | Weights | Configuration |
|---|---|---:|---|---|
| TMCAD | Classification | **83.88%** | [tmcad.pth](https://github.com/Qiang-Zou/BRT/releases/download/weights-v1/tmcad.pth) | [config](pretrained/tmcad/configs/finetune.json) |

The TMCAD result is 666/794 correct on the supplied test split. See the [model description](pretrained/tmcad/MODEL_CARD.md) for the training and evaluation settings.

## Installation and loading

Run these commands from the repository root:

```bash
python -m pip install -r pretrained/tmcad/requirements.txt
python pretrained/tmcad/download_weights.py
python pretrained/tmcad/load_model.py
```

The downloader saves `tmcad.pth` in `pretrained/tmcad/checkpoints/` and verifies its SHA-256. This tensor-only checkpoint uses the loading and testing entry points below; the original `classification.py` entry point expects a Lightning checkpoint.

## Testing

Prepare the matching feature caches described in [data preparation](pretrained/tmcad/data/README.md), then run:

```bash
python pretrained/tmcad/evaluate.py \
  --face-root /path/to/TMCAD_Tria/triangles \
  --topo-root /path/to/TMCAD_topo/brt \
  --output-dir outputs/tmcad_test
```

## Inference

```bash
python pretrained/tmcad/predict.py \
  --face /path/to/triangles/bearing/bearing_0788.bin \
  --topo /path/to/brt/bearing/bearing_0788.bin \
  --output outputs/prediction.json
```

Each model is paired with its own configuration and evaluation code under `pretrained/`.
