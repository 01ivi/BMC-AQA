# Method and implementation

This guide maps the supplied BMC-AQA manuscript to the feature-level reference
implementation. It does not describe the archived McMoE adapters.

## Notation and tensor contracts

The reference code fixes modality order to `(rgb, flow, audio)` and source order
to `(observed, generated, retrieved)`. Here `B` is batch size, `M=3`, `S=3`,
`T` is the cropped/padded temporal length, `d` is hidden width, and `N` is memory size.

| Tensor / output key | Shape | Meaning |
|---|---|---|
| Input `features[m]` | `[B,T,d_m]` | Extracted modality features |
| Input `observed` | `[B,M]` | Binary availability mask |
| Memory features | `[N,T,d_m]` | Complete features from the fitting training subset |
| Retrieval keys | `[B,d_k]`, `[N,d_k]` | L2-normalized mean/std MLP encodings |
| `RetrievalBatch.full_scores` | `[B,N]` | Mean cosine similarity across observed modalities |
| `RetrievalBatch.candidate_valid` | `[B,N]` | Eligible non-self entries |
| `RetrievalBatch.confidence_features` | `[B,3]` | Peak, top-two gap, normalized neighborhood entropy |
| `generated`, `retrieved` | `[B,M,T,d]` | Completion and encoded retrieval candidates |
| `generation_gate` | `[B,M,1,d]` | Channel-wise interpolation gate |
| `log_variance` | `[B,M]` | Completion log variance, bounded to `[-8,8]` |
| `source_prompts` | `[B,M,S,d]` | Modality + source + validity/confidence annotations |
| `source_validity`, `source_confidence` | `[B,M,S]` | Candidate eligibility and reliability |
| `calibrated_sources` | `[B,M,S,T,d]` | FCM-calibrated evidence |
| `source_weights` | `[B,M,S]` | Within-modality masked-softmax weights |
| `modality_weights` | `[B,M]` | Cross-modality masked-softmax weights |
| `fused` | `[B,T,d]` | Final fused sequence |
| `grade_probability` | `[B,G]` | Probability on ordered grade bins |
| `prediction` | `[B]` | Distribution expectation in `[0,1]` |

## Manuscript-to-code mapping

| Manuscript equation / mechanism | Code location | Implemented behavior |
|---|---|---|
| `retrieval_similarity` | `retrieval/memory.py: ModalityMemoryBank.retrieve` | Average observed-key cosine similarities; never read hidden query rows |
| `retrieval_aggregation` | `retrieval/memory.py: ModalityMemoryBank.retrieve` | Top-K softmax aggregation of complete training features for every modality |
| `retrieval_confidence` | `retrieval/memory.py`, `models/parag.py: RetrievalConfidence` | Peak/gap/entropy statistics and a sigmoid MLP; unavailable retrieval has zero confidence |
| `gated_generation` | `models/parag.py: GatedCompletion` | Retrieved encodings + learned seed; residual attention to zero-masked observed streams; channel gate and bounded log variance |
| `source_prompt` | `models/parag.py: SourcePrompts` | Sum modality/source embeddings and validity-confidence metadata MLP |
| `fcm_context` | `models/calibration.py: FeatureCalibration` | Joint average of valid pooled features + prompts |
| `source_calibration` | `models/calibration.py: FeatureCalibration` | Independent modality/source channel gates; `2 * validity * sigmoid(phi(context)) * source` |
| `source_moe` | `models/hmoe.py: HierarchicalMoE` | Three residual MLP experts shared cross-modally; per-modality source routers |
| `modality_moe` | `models/hmoe.py: HierarchicalMoE` | Independent RGB/flow/audio residual experts and reliability/prompt-conditioned modality router |
| Grade-based scoring | `models/score_head.py: GradeScoreHead` | Transformer decoder attends to all prompts and fused sequence; softmax expectation over fixed ordered bins |
| `completion_loss` | `losses.py: completion_loss` | Equal-weight modality means over artificially masked samples, including uncertainty term |
| `retrieval_loss` | `losses.py: retrieval_loss` | Soft-target cross-entropy over eligible non-self memory; scores supervise training only |
| `total_objective` | `losses.py: core_losses` | Task MSE + weighted completion and retrieval losses |

Paths are relative to `src/bmc_aqa/`. The labels above are manuscript LaTeX
equation labels, so the mapping remains valid if equation numbering changes.

## PARAG and source eligibility

The memory stores features before simulated missingness. Queries use only
observed modalities; training identifiers exclude every matching memory entry.
Scores do not influence neighbor ranking or aggregation.

Confidence statistics use the actual number of eligible selected neighbors.
For fewer than two neighbors, gap and entropy are zero. An empty memory or
self-only query produces zero retrieval evidence and zero retrieval confidence.

Generation uses `Q = encoded_retrieval + seed` and residual cross-attention to
the concatenated observed streams. Missing streams remain zero in that context.
The gate interpolates the refined query and encoded retrieval; `sigmoid(-nu)`
provides generation confidence. Both generated and retrieved evidence remain
separate candidates.

| Modality availability | Observed source | Generated source | Retrieved source |
|---|---:|---:|---:|
| Observed | eligible, confidence 1 | ineligible, confidence 0 | ineligible, confidence 0 |
| Missing, retrieval available | ineligible | eligible, confidence `sigmoid(-nu)` | eligible, learned retrieval confidence |
| Missing, retrieval unavailable | ineligible | eligible, context-only completion | ineligible |

Invalid prompts retain identity and encode zero validity/confidence. FCM
excludes them from its context; routers mask the associated evidence. The
grade decoder attends to every prompt, including those indicating invalidity.

## Calibration and hierarchical fusion

FCM uses every modality/source pair jointly. Its pooled context is the average
of `Pool(source) + prompt` across eligible sources. Each pair has its own
channel-gating MLP; invalid source representations are exactly zero after calibration.

Source experts have provenance-specific parameters shared across modalities.
Source-router inputs concatenate pooled calibrated sources, the three prompts,
candidate confidence, and the modality observation indicator. Only the observed
source is eligible for a visible modality; missing modalities mix generated and
retrieved candidates.

Modality experts use the same residual MLP architecture with independent
parameters. Their router takes pooled expert features, source-weighted prompts,
source-weighted reliability, and observation indicators. Normalized modality
weights fuse the temporal streams for scoring.

## Core objective

For each masked modality/sample, with `E = MSE(generated, stop_gradient(target))`:

```text
ell = (1 + eta/2 * exp(-nu)) * E + eta/2 * nu
L_comp = mean_over_active_modalities(mean_over_masked_samples(ell))
```

Complete targets are encoded with dropout disabled and gradients stopped.
The target path is called explicitly by training; the prediction forward path
never encodes hidden inputs for supervision. The log-variance objective may be
negative, as with other learned Gaussian variance objectives.

The retrieval target is `softmax(-abs(y_query - y_memory) / sigma_y)` over the
eligible pool, and the prediction is `softmax(similarity / tau_loss)` over the
same pool. Completion and retrieval losses over empty eligible sets are zero.
Training scores are normalized to `[0,1]` by the dataset task range.

No additional reconstruction, alignment, confidence-target, diversity, or
ranking objective is added to the three-loss core. Retrieval confidence and
fusion receive task gradients; generation variance receives completion gradients.

## Reference implementation choices

Raw extracted features are projected with linear layers and LayerNorm, then
encoded by modality-specific Transformers with sinusoidal positions. Exact
search re-encodes the memory with current key parameters; key encoding is
chunked to limit individual forward sizes, but full similarity search and its
training graph remain proportional to memory size.

FCM and HMoE can be independently removed with `use_fcm=false` or
`use_hmoe=false`. The latter replaces experts/routing with valid-source means
and equal modality averaging. These are reference ablations; a complete
baseline suite and a PARAG-removal experiment are not included in the core.

The architecture differs from archived checkpoints. The evaluator rejects
legacy checkpoint formats rather than partially loading incompatible weights.

