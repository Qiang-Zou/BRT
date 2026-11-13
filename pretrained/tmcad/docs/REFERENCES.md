# Release documentation references

Checked on 2026-11-13. These references supply the organization and interface conventions; all checkpoint details and scores in this release come from its own recorded experiment.

| Reference | Practice inspected | Applied here |
|---|---|---|
| [Point-MAE: model downloads](https://github.com/Pang-Yatian/Point-MAE#3-point-mae-models) | One table associates task, dataset, config, accuracy and release asset. Fine-tuning is explicitly named. | README model table, short checkpoint filenames, configuration links and GitHub Releases asset plan. |
| [Point-MAE: actual release assets](https://github.com/Pang-Yatian/Point-MAE/releases/tag/main) | The `main` release contains seven `.pth` files, including `modelnet_1k.pth` and `scan_hardest.pth`, plus `logs.tar`. README download links point directly to these assets. | Upload the model as a Release asset and use its direct download URL in the model table after the release is published. |
| [MAE: checkpoint downloads](https://github.com/facebookresearch/mae#fine-tuning-with-pre-trained-checkpoints) | README tables link to `.pth` files hosted at `dl.fbaipublicfiles.com`; detailed training commands are in separate documents. | Keep the front page focused on model downloads and usage; link detailed training and evaluation information separately. |
| [UV-Net: testing](https://github.com/AutodeskAILab/UV-Net#testing) | Explicit checkpoint argument and statement of validation-based checkpoint selection. | `evaluate.py --checkpoint ...`, documented selection criterion and fixed evaluation settings. |
| [BRepNet: evaluation on user data](https://github.com/AutodeskAILab/BRepNet#evaluating-the-segmentation-on-your-own-step-data) | Separate inference command and explanation of input/output files. | `predict.py --face ... --topo ...`, class/probability output and explicit cache requirements. |
| [PyTorch: saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) | Save model state, instantiate the architecture, load state and set evaluation mode. | Tensor-only `.pth`, strict `load_state_dict`, `weights_only=True` and `model.eval()`. |
| [GitHub: releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) | A tagged release groups notes with downloadable binary assets. | Source/asset separation, version tag, release notes and checksums. |

No entire third-party README was copied and no third-party performance claims were reused. Standard section names and command conventions were adapted to this package. Snapshot URLs and content hashes are recorded in [reference_sources.json](reference_sources.json).
