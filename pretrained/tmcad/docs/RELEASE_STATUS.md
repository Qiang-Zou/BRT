# Local release preparation

The checkpoint is included in the repository at [checkpoints/tmcad.pth](../checkpoints/tmcad.pth).

## Prepared artifacts

- README in English and Chinese, model card, setup, load, inference and evaluation commands.
- Tensor-only weight and SHA-256, class labels, relative data manifests, exclusions and recorded predictions.
- Frozen evaluation settings and a full independent packaging verification.
- Release note draft and separate source/weight assets.

## Items remaining before a public release

1. `download_weights.py` verifies the bundled file and uses the repository copy to restore it if missing. The separate `weights-v1` Release is optional.
2. The upstream BRT [LICENSE](https://github.com/Qiang-Zou/BRT/blob/main/LICENSE) contains GPL-3.0, while the [README copyright section](https://github.com/Qiang-Zou/BRT#1copyright) states restrictions on academic use and redistribution. Those statements are not aligned. Both sources have been inspected; this draft does not resolve the conflict or assert unrestricted redistribution rights.
3. No new license for the trained weight has been specified by its owner. This package retains upstream attribution and the upstream license text rather than assigning an unrelated MIT/Apache license.

The matching historical feature caches are also not publicly hosted by this package. That limits external replication of the exact benchmark score and must remain disclosed even if only the weights are released.
