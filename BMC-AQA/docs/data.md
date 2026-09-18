# Dataset preparation

BMC-AQA works on extracted features. Acquire them from the dataset authors'
upstream repositories:

- [PAMFN](https://github.com/qinghuannn/PAMFN): Fis-V and RG features and labels.
- [Skating-Mixer](https://github.com/AndyFrancesco29/Audio-Visual-Figure-Skating): FS1000 RGB/audio features and labels.
- [MCMoE dataset preparation](https://github.com/XuHuangbiao/MCMoE#dataset-preparation): FS1000 optical-flow features and the shared public layout.

Follow upstream download instructions and unpack features into the layout below.
This release does not download or redistribute dataset archives.

## Expected layout

```text
data/
├── FS1000/
│   ├── output_feature_fs1000_new/
│   │   └── <sample_id>.npy             # RGB: [T,d] or [T,views,d]
│   ├── i3d_avg_clip8_5s_fs1000/
│   │   └── <sample_id>.npy             # Optical flow: [T,d]
│   ├── ast_feature_fs1000_new/
│   │   └── <sample_id>.npy             # Audio: [T,d]
│   ├── train_fs1000_new.txt
│   └── val_fs1000_new.txt              # Official evaluation file
├── Fis-V/
│   ├── Fis-feature/
│   │   ├── FISV_rgb_VST.npy            # Dictionary: sample_id -> [T,d]
│   │   ├── FISV_flow_I3D.npy
│   │   └── FISV_audio_AST.npy
│   ├── train.txt
│   └── test.txt
└── RG/
    ├── RG-feature/
    │   ├── Ball_rgb_VST.npy            # One dictionary per apparatus/modality
    │   ├── Ball_flow_I3D.npy
    │   ├── Ball_audio_AST.npy
    │   ├── Clubs_rgb_VST.npy
    │   ├── Clubs_flow_I3D.npy
    │   ├── Clubs_audio_AST.npy
    │   ├── Hoop_rgb_VST.npy
    │   ├── Hoop_flow_I3D.npy
    │   ├── Hoop_audio_AST.npy
    │   ├── Ribbon_rgb_VST.npy
    │   ├── Ribbon_flow_I3D.npy
    │   └── Ribbon_audio_AST.npy
    ├── train.txt
    └── test.txt
```

Fis-V/RG `.npy` files contain pickled dictionaries, loaded with NumPy's
`allow_pickle=True` convention used by the public feature release. The FS1000
adapter averages the view axis for RGB arrays shaped `[T,views,d]`.

## Label files and normalization

The adapter follows the existing public label convention:

| Dataset | Text row, in order | Parsing |
|---|---|---|
| FS1000 | `sample_id TES PCS SS TR PE CO IN factor` | No header; select task column; PCS is divided by `factor` |
| Fis-V | `sample_id TES PCS` | Skip the first header line; select TES or PCS |
| RG | `sample_id Difficulty_Score Execution_Score Total_Score` | Skip the first header line; select apparatus from the ID prefix and predict Total_Score |

Task ranges used to normalize labels and restore prediction units:

| Dataset / task | Range |
|---|---:|
| FS1000 / TES | 130 |
| FS1000 / PCS (after factor correction) | 60 |
| FS1000 / SS, TR, PE, CO, IN | 10 |
| Fis-V / TES | 45 |
| Fis-V / PCS | 40 |
| RG / Total_Score | 25 |

Labels passed to training are `score / range`. Evaluation MSE uses predictions
multiplied by that range and labels in the same original units. For FS1000 PCS,
"original units" here refer to the factor-corrected score. These conventions
are inherited from the existing feature adapter, not newly estimated dataset maxima.

## Temporal features

Each modality is cropped or zero-padded to `clip_num` clips. Training uses a
random RGB-based crop start shared across the streams; memory and evaluation
use the deterministic center start. A shared start is clamped to each stream's
valid length when lengths differ. Input dimensions are inferred from the
selected dataset, so RGB, flow, and audio need not have equal feature widths.

The reference presets use 95 clips for FS1000, 124 for Fis-V, and 68 for RG.
The current core supports **simulated missingness of complete feature samples**.
All three feature files must be available in the training data to form the
memory and detached completion targets. Real missing training files require
an additional availability-aware data adapter.

## Custom paths

The `[data]` section can override any default directory or label file. Paths
are interpreted relative to `--data-root`; absolute paths are also accepted:

```toml
[data]
dataset = "FS1000"
action_type = "TES"
clip_num = 95
rgb_path = "features/rgb"
flow_path = "features/flow"
audio_path = "features/audio"
train_label_path = "splits/train.txt"
test_label_path = "splits/test.txt"
```

The CLI spells Fis-V as `FisV`, while its default data directory is `Fis-V`.
RG IDs should start with their apparatus followed by an underscore, such as
`Ball_001`. Each training identifier must be unique and evaluation identifiers
must be disjoint from both the fitting and validation subsets.

