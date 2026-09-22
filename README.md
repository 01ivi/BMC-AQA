# BMC-AQA: HIERARCHICAL MOE WITH PROMPT-ANNOTATED RAG FOR MULTIMODAL ACTION QUALITY ASSESSMENT UNDER IMBALANCED MODALITY MISSINGNESS



## Repository structure

```text
BMC-AQA/
├── README.md                    # Method, repository guide, and runnable commands
├── pyproject.toml               # Installable bmc_aqa package
├── requirements.txt             # Runtime dependencies
├── .gitignore
├── configs/                     # Explicit reference presets, in paper modality order
│   ├── fs1000.toml              # FS1000 / TES
│   ├── fisv.toml                # Fis-V / TES
│   └── rg.toml                  # RG / Ball
├── docs/
│   ├── method.md                # Equation-to-code mapping and tensor contracts
│   ├── data.md                  # Feature layouts, labels, and score normalization
│   └── experiments.md           # IMR protocol, artifacts, and evaluation boundaries
├── scripts/
│   ├── train.py                 # Train with a training-set validation holdout
│   ├── evaluate.py              # Full, IMR, and fixed-subset evaluation
│   ├── run_imr.sh               # Six independently trained IMR configurations
│   └── smoke_test.py            # Synthetic retrieval / forward / backward check
├── src/bmc_aqa/
│   ├── __init__.py              # Public BMCAQA and memory-bank API
│   ├── constants.py             # Modality and source axis conventions
│   ├── config.py                # TOML loading and JSON export
│   ├── data.py                  # FS1000, Fis-V, and RG feature adapters
│   ├── engine.py                # Training, checkpoint loading, and evaluation
│   ├── losses.py                # Three manuscript core losses
│   ├── protocol.py              # Missingness masks, holdout, and AQA metrics
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── memory.py            # Observed-only cosine search and weighted evidence
│   └── models/
│       ├── __init__.py
│       ├── bmc_aqa.py           # PARAG → FCM → HMoE → scoring composition
│       ├── encoders.py          # Modality encoders and normalized mean/std keys
│       ├── parag.py             # Confidence, cross-attention generation, prompts
│       ├── calibration.py       # Valid-source context and channel gates
│       ├── hmoe.py              # Source experts/router and modality experts/router
│       └── score_head.py        # Grade queries, distribution, score expectation
├── tests/
│   ├── test_core.py             # Equations, masks, routing, gradients, hidden inputs
│   ├── test_data.py             # Public dataset adapter formats
└── └── test_cli.py              # Tiny train / save / reload / evaluate integration
```

`data/`, `runs/`, and checkpoints are local artifacts ignored by Git. The
[archive](legacy/README.md) retains earlier experiment code and uses a different
modality order and checkpoint format.

## Datasets

The implementation consumes **pre-extracted temporal features**; video/audio
backbone training and feature extraction are outside this release.

| Dataset | CLI identifier | Supported tasks | Feature/label sources |
|---|---|---|---|
| FS1000 | `FS1000` | `TES`, `PCS`, `SS`, `TR`, `PE`, `CO`, `IN` | RGB/audio and labels: [Skating-Mixer](https://github.com/AndyFrancesco29/Audio-Visual-Figure-Skating); flow: [MCMoE](https://github.com/XuHuangbiao/MCMoE#dataset-preparation) |
| Fis-V | `FisV` | `TES`, `PCS` | [PAMFN](https://github.com/qinghuannn/PAMFN) |
| Rhythmic Gymnastics | `RG` | `Ball`, `Clubs`, `Hoop`, `Ribbon` | [PAMFN](https://github.com/qinghuannn/PAMFN) |

Prepare the public feature layout under `data/`. Full directory trees, label
formats, task-specific score ranges, and custom-path settings are documented
in [docs/data.md](docs/data.md). Data and upstream model weights are not
redistributed here.

## Training

Train FS1000 TES with missing rates `(0.3, 0.5, 0.7)` in RGB/flow/audio order:

```bash
python scripts/train.py \
  --config configs/fs1000.toml \
  --data-root data \
  --imr-rates 0.3 0.5 0.7 \
  --output-dir runs/fs1000_tes/rgb0.3_flow0.5_audio0.7
```

Fis-V and RG use the same entry point:

```bash
python scripts/train.py --config configs/fisv.toml --data-root data --output-dir runs/fisv_tes
python scripts/train.py --config configs/rg.toml --data-root data --output-dir runs/rg_ball
```

Change tasks with `--action-type PCS` or `--action-type Ribbon`, as appropriate
for the selected dataset. CLI overrides also support `--epochs`, `--batch-size`,
`--seed`, and `--num-workers`. Model and loss settings are explicit in TOML.

Training holds out a score-stratified subset of the training split for model
selection. **Memory contains only the remaining training samples, before
simulated masking; training queries exclude all self matches.** Checkpoints are
selected by validation MSE and Spearman, and the official evaluation split is
used only by the evaluation command. See [the protocol](docs/experiments.md).

The objective is:

```text
L_core = L_task + lambda_comp * L_comp + lambda_ret * L_ret
```

Completion uses detached complete-feature targets and bounded log variance;
its average gives each artificially masked modality equal weight. Retrieval
uses normalized training scores for soft-target supervision. Scores are never
inputs to inference search or prediction.

## Evaluation

Evaluate the full input and the checkpoint's training IMR configuration:

```bash
python scripts/evaluate.py \
  --checkpoint runs/fs1000_tes/rgb0.3_flow0.5_audio0.7/best_mse.pt \
  --data-root data \
  --protocol imr --mask-seed 2027 \
  --output-dir runs/fs1000_tes/rgb0.3_flow0.5_audio0.7/test
```

The evaluator loads `memory.pt` beside the checkpoint and checks its training
identifiers. Keep this bank when moving a run; rebuilding it with the holdout
or evaluation samples changes the experiment.

| Protocol | Evaluation |
|---|---|
| `imr` | Full modalities + one IMR setting; optionally override with `--imr-rates` |
| `all-imr` | Full modalities + all six permutations of `(0.3, 0.5, 0.7)` for one checkpoint |
| `fixed` | All seven non-empty modality subsets + an incomplete-subset average |

For six independently trained and evaluated IMR models:

```bash
bash scripts/run_imr.sh configs/fs1000.toml data runs/fs1000_tes_imr auto
```

`all-imr` measures one model under six missingness settings; `run_imr.sh`
trains a separate model for each setting. Choose the protocol matching the
experiment being reported.

Evaluation writes `metrics.csv`, `metrics.md`, `metrics.json`, per-sample
`predictions.csv` (including masks), and `evaluation.json`. **MSE is reported
on the original task score scale.** Spearman averages use Fisher-Z; MSE averages
use arithmetic means. Undefined Spearman values are recorded as `null`/`n/a`.


