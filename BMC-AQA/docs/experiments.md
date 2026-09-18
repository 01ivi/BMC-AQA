# Experiment protocol and artifacts

The TOML files are runnable reference presets. This snapshot has not been
trained on the complete public datasets or verified against the supplied
manuscript tables. Report such a reproduction only after running and checking
the final experiment settings.

The core smoke test and 15 automated checks were verified on CPU with
Python 3.12.14, PyTorch 2.14.0, and NumPy 2.5.3. The dependency minimums
in `pyproject.toml` are not a tested compatibility matrix.

## Missingness protocol

Every missing-rate triple is ordered `(RGB, optical flow, audio)`:

| Setting | RGB | Flow | Audio |
|---|---:|---:|---:|
| 1 | 0.3 | 0.5 | 0.7 |
| 2 | 0.3 | 0.7 | 0.5 |
| 3 | 0.5 | 0.3 | 0.7 |
| 4 | 0.5 | 0.7 | 0.3 |
| 5 | 0.7 | 0.3 | 0.5 |
| 6 | 0.7 | 0.5 | 0.3 |

Masks are sampled independently per sample and modality. Rejection sampling
redraws samples with no observations. The specified rates are Bernoulli rates
before conditioning, so empirical marginal missingness after rejection may
be lower. A full-modality sample is allowed.

Validation and final evaluation generate the complete mask tensor before
batching, using a dedicated CPU generator. Thus changing evaluation batch size
does not change sample masks. The presets select checkpoints with mask seed
2026; the example final-test commands and six-setting launcher use seed 2027.
These are example protocol choices, not asserted manuscript seeds.

## Data boundaries and model selection

1. Parse the training labels and create a seeded, score-stratified validation
   holdout (default 15%).
2. Use only the fitting subset for gradient updates and the deterministic
   feature memory. Keep complete features before simulated masking.
3. Exclude every self identifier during training retrieval. Labels create
   retrieval soft targets; similarity search never uses them.
4. Select `best_mse.pt` and, when defined, `best_spearman.pt` using the fixed
   validation IMR masks. Training does not open official test feature files.
5. Evaluate the chosen checkpoint on the official evaluation split with the
   same training bank. The evaluator rejects identifier overlap with fitting
   or validation samples and rejects a different memory ID list.

The implementation does not refit on the holdout after selection. Its final
bank stays restricted to the fitting training subset. Keep this convention
when comparing with another method, or explicitly describe a different
refitting protocol.

## Training artifacts

```text
runs/<experiment>/
├── config.json                   # Effective TOML configuration + CLI overrides
├── split.json                    # Fitting/validation IDs, dimensions, runtime versions
├── memory.pt                     # Complete fitting-subset features, IDs, normalized scores
├── history.json                  # Epoch losses and validation metrics
├── last.pt                       # Final epoch
├── best_mse.pt                   # Best validation MSE
├── best_spearman.pt               # Best defined validation Spearman (may be absent)
├── metrics.csv                   # Final epoch validation metrics
├── metrics.json
└── metrics.md
```

Checkpoints contain the architecture configuration, input dimensions, model
state, epoch, validation result, score range, and the memory/holdout IDs. They
are inference snapshots; optimizer state and exact training resume are outside
this reference release. Existing run directories with checkpoints or a memory
bank are not overwritten.

Move `memory.pt` with the checkpoint. `--memory-path` supports a relocated bank
with the same saved training identifiers. Do not regenerate it with different
crops or features: identifier checks do not prove feature-content equality.

## Evaluation modes

```bash
# One checkpoint: complete input and one IMR setting
python scripts/evaluate.py --checkpoint runs/example/best_mse.pt \
  --data-root data --protocol imr --mask-seed 2027 --output-dir runs/example/test

# One checkpoint: complete input and all six IMR settings
python scripts/evaluate.py --checkpoint runs/example/best_mse.pt \
  --data-root data --protocol all-imr --mask-seed 2027 --output-dir runs/example/all_imr

# One checkpoint: the seven fixed non-empty modality subsets
python scripts/evaluate.py --checkpoint runs/example/best_mse.pt \
  --data-root data --protocol fixed --output-dir runs/example/fixed

# Independent training and matching IMR testing for all six settings
bash scripts/run_imr.sh configs/fs1000.toml data runs/fs1000_tes_imr auto
```

Each evaluation directory contains:

| File | Contents |
|---|---|
| `metrics.csv`, `metrics.json`, `metrics.md` | Spearman and raw-scale MSE by input setting |
| `predictions.csv` | Per-sample predictions, labels, identifiers, and the exact modality mask |
| `evaluation.json` | Checkpoint/bank paths, checkpoint epoch, protocol, seed, modality order, score range, sample count |

The fixed-subset incomplete average excludes the full-modality row. Spearman
uses Fisher-Z averaging; MSE uses arithmetic averaging. Undefined Spearman is
omitted from the correlation mean but its MSE remains included in the MSE mean.
The core evaluator reports one task at a time. For multi-task tables, aggregate
task rows with `bmc_aqa.protocol.aggregate_metrics` using the paper's specified
task scope.

## What the checks establish

Synthetic tests check manual retrieval statistics, the two softmax
normalizations, loss equations, modality balancing, detached targets, invalid
prompt exclusion from FCM, source/modality routing, finite gradients, and
invariance to hidden query features or changed memory labels at inference.
The CLI integration test creates temporary features, trains one CPU epoch,
loads the checkpoint and bank, evaluates IMR/fixed masks, and rejects a changed
memory identifier list.

These checks establish the reference implementation's behavior. They do not
establish trained performance, CUDA determinism, or reproduction of reported
benchmark numbers. Citation metadata, the authors' chosen license, trained
weights, and the final verified benchmark settings remain to be supplied with
the publication release.
