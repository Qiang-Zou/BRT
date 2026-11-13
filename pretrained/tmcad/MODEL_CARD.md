# Model card: BRT TMCAD fine-tuned classifier

## Model details

- Task: single-label classification of CAD parts into ten historical TMCAD categories.
- Architecture: the bundled historical BRT implementation; `BRTClassification(num_classes=10, num_control_pts=28)`.
- Parameters: 2,292,874. Weight format: tensor-only PyTorch `state_dict`.
- Class mapping: [data/class_names.json](data/class_names.json).
- Intended use: research evaluation and inference on compatible preprocessed CAD features.
- Model identity, hashes and environment: [model_card.json](model_card.json).

## Evaluation

| Model | Test set / protocol | ACC |
|---|---|---:|
| This fine-tuned checkpoint | Recorded historical split, 794 admitted CADs | 83.8791% (666/794) |
| BRT paper | Author experiment | 83.45% |

The rows are not a controlled, same-split comparison. The positive numeric difference is not evidence of outperforming the paper. There is one evaluated run, not a multi-seed mean or uncertainty estimate.

The reported checkpoint was chosen using minimum validation loss, not test accuracy. The original test and the independent packaging verification used the same fixed checkpoint. Packaging verification is not another model-selection experiment.

The recorded evaluator uses seed 0, batch size 16, eight data-loader workers, eight CPU threads, FP32 highest precision, the original sample order and `drop_last=False`. The historical wire encoder initializes recurrent hidden states randomly even in evaluation mode. Single-sample predictions or different batching can therefore differ from the recorded batch evaluation.

The package was verified from a relocated directory with the original project's `PYTHONPATH` unset, using Python 3.10.21, PyTorch 2.2.1+cu121, NumPy 1.26.4 and tqdm 4.67.1. All 794 predictions matched the original evaluation. Installation into a fresh environment has not been tested.

## Data

| Split | Entries before loader admission | Admitted |
|---|---:|---:|
| Train | 4,780 | 3,708 |
| Validation | 1,024 | 789 |
| Test | 1,026 | 794 |

The original order and labels are preserved in relative-path manifests. Exclusions are listed in [data/admission.json](data/admission.json). The split is historical; an author-exact match has not been established. Content-level duplicate leakage has not been audited for this historical split. A separate audit of a different, larger TMCAD split must not be attributed to this one.

## Training provenance

The supervised initial checkpoint was selected at epoch 24 of a historical run that eventually completed 1,000 epochs. Its recorded Adam configuration was learning rate `1e-4`, betas `(0.9, 0.999)`. The complete original launch command and training seed are not established by available evidence.

Additional fine-tuning used seed 0, FP32 highest, Adam `1e-5`, betas `(0.001, 0.999)`, batch 16, a fresh optimizer and all parameters trainable. No extra masking or scheduler was added. Maximum length was 30 epochs with patience 8; it stopped after epoch 9 and selected epoch 1. Initial and selected checkpoints are bound to hashes in the machine-readable metadata.

## Limitations

The historical model retains a four-head face encoder and degree-10 edge input features, unlike the corresponding paper description. Known legacy geometry, ordering and topology behavior is preserved to match the evaluated weight. This release does not claim those issues were corrected.

The matching processed caches are not distributed here, and a verified cache reconstruction from raw STEP files is unavailable. Loading the checkpoint and producing predictions is supported; independently recovering the 83.88% metric requires those matching inputs. New CAD geometry must undergo compatible preprocessing. The model has not been validated for different dataset versions, changed labels, geometric robustness or deployment outside this research setting.

## Source and license

BRT is attributed to Qiang Zou and Lizhen Zhu. Numerical inference operations are preserved; only unused imports and the unused Lightning training wrapper were removed for the inference package. [NOTICE.md](NOTICE.md) records those modifications. [RELEASE_STATUS.md](docs/RELEASE_STATUS.md) records the unresolved upstream license-text inconsistency and publication status.
