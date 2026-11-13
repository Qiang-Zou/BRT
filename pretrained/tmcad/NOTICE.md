# Attribution and changes

BRT was developed by Lizhen Zhu and Qiang Zou. Source repository: https://github.com/Qiang-Zou/BRT . Paper: https://arxiv.org/abs/2504.07134 . The original implementation is not attributed to the checkpoint packager.

The numerical runtime comes from the historical implementation that produced the evaluated checkpoint, not an asserted unmodified snapshot of current upstream main. File hashes are in `model_card.json`; the original fine-tuning source hashes are in `configs/finetune.json`.

Packaging modifications, 2026-11-13:

- The historical dataset adapter had already removed two unused DGL imports.
- The inference export removes the unused Lightning classification training wrapper and its imports. The `BRTClassification` class is AST-identical to the evaluated version.
- Portable loading, download, single-model prediction and evaluation wrappers were added. Default evaluation retains the recorded seed, batch, order, precision and sample-admission behavior.
- The exported weight contains tensors only, exactly equal to the selected full checkpoint's model tensors.

The upstream license text is retained in `LICENSE.upstream`. Its README contains additional, inconsistent redistribution restrictions, recorded in `docs/RELEASE_STATUS.md`. This local draft does not assign a new weight license or claim that the upstream inconsistency has been resolved.
