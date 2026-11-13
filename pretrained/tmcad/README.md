# BRT TMCAD

[简体中文](README.zh-CN.md) · [Model details](MODEL_CARD.md) · [Data preparation](data/README.md)

Model weights and evaluation code for TMCAD part classification, based on [BRT](https://github.com/Qiang-Zou/BRT).

## Models

| Task | Dataset | Config | ACC | Download |
|---|---|---|---:|---|
| Classification | TMCAD, 10 classes | [Model](configs/model.json) / [Fine-tuning](configs/finetune.json) | **83.88%** | [Checkpoint](checkpoints/tmcad.pth?raw=true) |

ACC is measured on the supplied test split: **666/794** correct. The checkpoint is a PyTorch `state_dict` (13.4 MB). See [model details](MODEL_CARD.md) for training, data and evaluation settings.

## Installation

Run the following commands from `pretrained/tmcad/`. Commands to run from the repository root are in [MODELS.md](../../MODELS.md).

Tested with Python 3.10 and PyTorch 2.2.1 on Linux.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Verify and load the checkpoint included in `checkpoints/`:

```bash
python download_weights.py
python load_model.py
```

## Testing

Prepare the matching face and topology feature caches following [data/README.md](data/README.md), then run:

```bash
python evaluate.py \
  --checkpoint checkpoints/tmcad.pth \
  --face-root /path/to/TMCAD_Tria/triangles \
  --topo-root /path/to/TMCAD_topo/brt \
  --output-dir outputs/tmcad_test
```

The evaluator uses the recorded CPU protocol and checks the test split. Use a new output directory for each run. Expected output:

```json
{"num_samples": 794, "correct": 666, "accuracy": 83.87909319899245}
```

The [test result](evidence/test_result.json) and [verification record](evidence/release_verification.json) are included.

## Inference

Predict the class of one CAD model from its matching preprocessed feature files:

```bash
python predict.py \
  --face /path/to/triangles/bearing/bearing_0788.bin \
  --topo /path/to/brt/bearing/bearing_0788.bin \
  --output outputs/prediction.json
```

The output contains the class ID, class name and probabilities.

## Citation and license

Please cite the original [BRT paper](https://doi.org/10.1016/j.cad.2025.103940); see [citation.bib](citation.bib). Attribution and license information are in [NOTICE.md](NOTICE.md) and [LICENSE.upstream](LICENSE.upstream).
