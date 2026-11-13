# BRT TMCAD

[English](README.md) · [模型说明](MODEL_CARD.md) · [数据准备](data/README.md)

基于 [BRT](https://github.com/Qiang-Zou/BRT) 的 TMCAD 零件分类权重与评测代码。

## 模型权重

| 任务 | 数据集 | 配置 | ACC | 下载 |
|---|---|---|---:|---|
| 零件分类 | TMCAD，10 类 | [模型](configs/model.json) / [微调](configs/finetune.json) | **83.88%** | [权重](https://github.com/Qiang-Zou/BRT/releases/download/weights-v1/tmcad.pth) |

ACC 在随包提供的测试划分上测得，预测正确 **666/794**。权重采用 PyTorch `state_dict` 格式，大小约 13.4 MB。训练、数据与评测设置见[模型说明](MODEL_CARD.md)。

## 安装

以下命令在 `pretrained/tmcad/` 目录运行。在仓库根目录运行的命令见 [MODELS.md](../../MODELS.md)。

已验证环境：Linux、Python 3.10、PyTorch 2.2.1。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

将权重放入 `checkpoints/`，校验并加载：

```bash
python download_weights.py
python load_model.py
```

## 测试

按照[数据说明](data/README.md)准备匹配的面特征与拓扑缓存，然后运行：

```bash
python evaluate.py \
  --checkpoint checkpoints/tmcad.pth \
  --face-root /path/to/TMCAD_Tria/triangles \
  --topo-root /path/to/TMCAD_topo/brt \
  --output-dir outputs/tmcad_test
```

脚本使用记录中的 CPU 评测设置并核查测试划分。每次运行使用新的输出目录。预期输出：

```json
{"num_samples": 794, "correct": 666, "accuracy": 83.87909319899245}
```

包内附有[测试结果](evidence/test_result.json)和[复测记录](evidence/release_verification.json)。

## 单模型预测

输入同一 CAD 模型预处理后的面特征与拓扑文件：

```bash
python predict.py \
  --face /path/to/triangles/bearing/bearing_0788.bin \
  --topo /path/to/brt/bearing/bearing_0788.bin \
  --output outputs/prediction.json
```

输出包含类别编号、类别名称及各类别概率。

## 引用与许可

请引用原始 [BRT 论文](https://doi.org/10.1016/j.cad.2025.103940)，条目见 [citation.bib](citation.bib)。代码归属与许可说明见 [NOTICE.md](NOTICE.md) 和 [LICENSE.upstream](LICENSE.upstream)。
