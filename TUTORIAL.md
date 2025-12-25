# PIDNet Step-by-Step Tutorial

## Table of Contents
1. [Introduction](#1-introduction)
2. [Understanding PIDNet Architecture](#2-understanding-pidnet-architecture)
3. [Installation and Setup](#3-installation-and-setup)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Training Your First Model](#5-training-your-first-model)
6. [Evaluating Model Performance](#6-evaluating-model-performance)
7. [Running Inference on Custom Images](#7-running-inference-on-custom-images)
8. [Understanding the Code Structure](#8-understanding-the-code-structure)
9. [Advanced Topics](#9-advanced-topics)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Introduction

### What is PIDNet?

PIDNet (Proportional-Integral-Derivative Network) is a real-time semantic segmentation network inspired by the PID controller from control theory. It achieves state-of-the-art performance on benchmark datasets while maintaining high inference speeds, making it ideal for real-time applications like autonomous vehicles and medical imaging.

### Key Features

- **Real-time Performance**: 93.2 FPS on Cityscapes (PIDNet-S) and 153.7 FPS on CamVid (PIDNet-S)
- **High Accuracy**: 78.6% mIOU on Cityscapes test set (PIDNet-S), 80.6% mIOU (PIDNet-L)
- **Novel Architecture**: Three-branch design (P, I, D) mimicking PID controller behavior
- **Practical Applications**: Directly applicable to autonomous driving, robotics, and medical imaging

### Model Variants

- **PIDNet-S (Small)**: Fast and lightweight (32 base channels, m=2, n=3)
- **PIDNet-M (Medium)**: Balanced speed and accuracy (64 base channels, m=2, n=3)
- **PIDNet-L (Large)**: Maximum accuracy (64 base channels, m=3, n=4)

---

## 2. Understanding PIDNet Architecture

### The PID Controller Analogy

PIDNet draws inspiration from the classic PID controller in control systems:

- **P (Proportional)**: Responds to current error - Detail preservation branch
- **I (Integral)**: Accounts for past errors - Context embedding branch  
- **D (Derivative)**: Predicts future errors - Boundary detection branch

### Three-Branch Architecture

```
Input Image (3 × H × W)
         │
         ├─────────────────┬─────────────────┐
         │                 │                 │
    P Branch          I Branch          D Branch
  (Detail)          (Context)         (Boundary)
         │                 │                 │
         └─────────┬───────┴─────────────────┘
                   │
              Final Fusion
                   │
            Segmentation Map
```

#### 1. I-Branch (Integral/Context)
- **Purpose**: Captures semantic context and global information
- **Structure**: Standard CNN backbone with multiple downsampling layers
- **Layers**: conv1 → layer1 → layer2 → layer3 → layer4 → layer5
- **Output**: High-level semantic features at 1/32 resolution

#### 2. P-Branch (Proportional/Detail)
- **Purpose**: Preserves fine spatial details and boundary information
- **Structure**: Parallel lightweight branch starting from layer3
- **Key Components**: 
  - PagFM (Pixel-attention-guided Fusion Module) for feature aggregation
  - Compression layers to reduce channel dimensions
- **Output**: Detail-rich features at 1/8 resolution

#### 3. D-Branch (Derivative/Boundary)
- **Purpose**: Explicitly detects boundaries between objects
- **Structure**: Separate branch for boundary detection
- **Key Components**:
  - Difference operators (diff3, diff4) to capture boundary signals
  - Light_Bag/Bag modules for boundary refinement
- **Output**: Binary boundary maps

### Key Modules

#### PagFM (Pixel-attention-guided Fusion Module)
Fuses features from P-branch and I-branch using pixel-wise attention:
```python
# Guides detail preservation with semantic context
x_p = PagFM(x_detail, x_context)
```

#### PAPPM/DAPPM (Pyramid Attention Pooling Module)
Captures multi-scale context information:
- **PAPPM**: Used in PIDNet-S/M (lightweight)
- **DAPPM**: Used in PIDNet-L (more powerful)

#### Bag/Light_Bag (Bilateral Aggregation Gate)
Aggregates features from all three branches:
```python
# Final fusion of P, I, D branches
output = Bag(x_p, x_i, x_d)
```

---

## 3. Installation and Setup

### Step 1: System Requirements

**Hardware Requirements:**
- GPU: NVIDIA GPU with CUDA support (RTX 3090 recommended for speed benchmarks)
- RAM: 16GB minimum, 32GB recommended
- Storage: 50GB for datasets and models

**Software Requirements:**
- Python: 3.7 or higher
- CUDA: 10.2 or higher
- cuDNN: Compatible with CUDA version

### Step 2: Clone the Repository

```bash
# Clone the PIDNet repository
git clone https://github.com/XuJiacong/PIDNet.git
cd PIDNet
```

### Step 3: Create Python Environment

```bash
# Create a virtual environment (recommended)
conda create -n pidnet python=3.8
conda activate pidnet

# OR using venv
python -m venv pidnet_env
source pidnet_env/bin/activate  # Linux/Mac
# pidnet_env\Scripts\activate  # Windows
```

### Step 4: Install Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
# For CUDA 11.3:
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# For CUDA 11.7:
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Install other required packages
pip install opencv-python
pip install tensorboardX
pip install pyyaml
pip install scipy
pip install easydict
pip install Pillow
pip install tqdm
```

### Step 5: Verify Installation

```bash
# Test PyTorch and CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch version: 1.10.0+cu113
CUDA available: True
CUDA version: 11.3
```

---

## 4. Dataset Preparation

### Option A: Cityscapes Dataset

#### Step 1: Download Cityscapes

1. Register at [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
2. Download the following packages:
   - `leftImg8bit_trainvaltest.zip` (11GB) - Images
   - `gtFine_trainvaltest.zip` (241MB) - Fine annotations

#### Step 2: Extract and Organize

```bash
# Create data directory
mkdir -p data/cityscapes

# Extract datasets
unzip leftImg8bit_trainvaltest.zip -d data/cityscapes/
unzip gtFine_trainvaltest.zip -d data/cityscapes/

# Expected directory structure:
# data/cityscapes/
# ├── leftImg8bit/
# │   ├── train/
# │   ├── val/
# │   └── test/
# └── gtFine/
#     ├── train/
#     ├── val/
#     └── test/
```

#### Step 3: Verify Data Lists

The repository includes pre-generated data lists in `data/list/cityscapes/`:
- `train.lst`: Training image paths
- `val.lst`: Validation image paths
- `test.lst`: Test image paths

Check that paths in these files match your data directory structure.

### Option B: CamVid Dataset

#### Step 1: Download CamVid

1. Download from [Kaggle CamVid Dataset](https://www.kaggle.com/datasets/carlolepelaars/camvid)
2. The resolution should be 960×720 (original)

#### Step 2: Organize Dataset

```bash
# Create data directory
mkdir -p data/camvid/images
mkdir -p data/camvid/labels

# Move all images to images/ directory
# Move all colored labels to labels/ directory

# Expected structure:
# data/camvid/
# ├── images/
# │   ├── 0001TP_006690.png
# │   ├── 0001TP_006720.png
# │   └── ...
# └── labels/
#     ├── 0001TP_006690_L.png
#     ├── 0001TP_006720_L.png
#     └── ...
```

#### Step 3: Dataset Lists

Pre-generated lists are in `data/list/camvid/`:
- `train.lst`: Training split (from SegNet-Tutorial)
- `val.lst`: Validation split
- `test.lst`: Test split

---

## 5. Training Your First Model

### Step 1: Download Pretrained Weights

Download ImageNet pretrained models for initialization:

**PIDNet-S:**
```bash
mkdir -p pretrained_models/imagenet
# Download from: https://drive.google.com/file/d/1hIBp_8maRr60-B3PF0NVtaA6TYBvO4y-/view
# Save as: pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar
```

**PIDNet-M:**
```bash
# Download from: https://drive.google.com/file/d/1gB9RxYVbdwi9eO5lbT073q-vRoncpYT1/view
# Save as: pretrained_models/imagenet/PIDNet_M_ImageNet.pth.tar
```

**PIDNet-L:**
```bash
# Download from: https://drive.google.com/file/d/1Eg6BwEsnu3AkKLO8lrKsoZ8AOEb2KZHY/view
# Save as: pretrained_models/imagenet/PIDNet_L_ImageNet.pth.tar
```

### Step 2: Understanding Configuration Files

Configuration files are in `configs/` directory. Let's look at `configs/cityscapes/pidnet_small_cityscapes.yaml`:

```yaml
GPUS: (0,1)                    # GPU IDs to use
DATASET:
  DATASET: cityscapes          # Dataset name
  NUM_CLASSES: 19              # Number of classes
  
MODEL:
  NAME: pidnet_small           # Model variant
  PRETRAINED: "pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar"

TRAIN:
  BATCH_SIZE_PER_GPU: 6        # Batch size per GPU
  END_EPOCH: 484               # Total training epochs
  LR: 0.01                     # Learning rate
  OPTIMIZER: sgd               # Optimizer type
```

### Step 3: Start Training

#### Train PIDNet-S on Cityscapes (2 GPUs)

```bash
python tools/train.py \
  --cfg configs/cityscapes/pidnet_small_cityscapes.yaml \
  GPUS (0,1) \
  TRAIN.BATCH_SIZE_PER_GPU 6
```

#### Train PIDNet-M on Single GPU

```bash
python tools/train.py \
  --cfg configs/cityscapes/pidnet_medium_cityscapes.yaml \
  GPUS (0,) \
  TRAIN.BATCH_SIZE_PER_GPU 4
```

#### Train PIDNet-L with Train+Val Sets (4 GPUs)

```bash
python tools/train.py \
  --cfg configs/cityscapes/pidnet_large_cityscapes_trainval.yaml \
  GPUS (0,1,2,3) \
  TRAIN.BATCH_SIZE_PER_GPU 3
```

### Step 4: Monitor Training

Training outputs are saved to:
- **Checkpoints**: `output/cityscapes/pidnet_small/`
- **Logs**: `log/cityscapes/pidnet_small/`
- **TensorBoard**: `log/cityscapes/pidnet_small/`

Monitor with TensorBoard:
```bash
tensorboard --logdir=log/cityscapes/pidnet_small/
```

### Step 5: Understanding Training Output

```
Epoch: [1][10/500]  Time 2.341 (2.450)  Loss 2.1234 (2.2341)
Epoch: [1][20/500]  Time 2.301 (2.420)  Loss 2.0123 (2.1523)
...
```

- **Time**: Batch processing time (current/average)
- **Loss**: Training loss (current/average)
- **Epoch**: Current epoch and batch numbers

### Training Tips

1. **Batch Size**: Adjust based on GPU memory
   - RTX 3090 (24GB): 6-8 per GPU for PIDNet-S
   - RTX 3080 (10GB): 3-4 per GPU for PIDNet-S

2. **Learning Rate**: Default 0.01 works well, but you can adjust:
   ```bash
   TRAIN.LR 0.005  # Lower LR for fine-tuning
   ```

3. **Resume Training**: If interrupted, add:
   ```bash
   TRAIN.RESUME true
   ```

4. **Multi-Scale Training**: Enabled by default for better accuracy
   ```yaml
   TRAIN.MULTI_SCALE: true
   TRAIN.FLIP: true
   ```

---

## 6. Evaluating Model Performance

### Step 1: Download Pretrained Models

For quick evaluation, download pre-trained models:

**Cityscapes Models:**
```bash
mkdir -p pretrained_models/cityscapes

# PIDNet-S (Val: 78.8% mIOU)
# Download: https://drive.google.com/file/d/1JakgBam_GrzyUMp-NbEVVBPEIXLSCssH/view
# Save as: pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt

# PIDNet-M (Val: 79.9% mIOU)
# Download: https://drive.google.com/file/d/1q0i4fVWmO7tpBKq_eOyIXe-mRf_hIS7q/view
# Save as: pretrained_models/cityscapes/PIDNet_M_Cityscapes_val.pt

# PIDNet-L (Val: 80.9% mIOU)
# Download: https://drive.google.com/file/d/1AR8LHC3613EKwG23JdApfTGsyOAcH0_L/view
# Save as: pretrained_models/cityscapes/PIDNet_L_Cityscapes_val.pt
```

**CamVid Models:**
```bash
mkdir -p pretrained_models/camvid

# PIDNet-S (Test: 80.1% mIOU)
# Download: https://drive.google.com/file/d/1h3IaUpssCnTWHiPEUkv-VgFmj86FkY3J/view
# Save as: pretrained_models/camvid/PIDNet_S_Camvid_Test.pt
```

### Step 2: Evaluate on Validation Set

#### Cityscapes Validation

```bash
python tools/eval.py \
  --cfg configs/cityscapes/pidnet_small_cityscapes.yaml \
  TEST.MODEL_FILE pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt
```

Expected output:
```
=> loaded pretrained model 'pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt'
[EVAL] Loss: 0.5234, mIoU: 78.8%, Acc: 96.5%
Per-class IoU:
  road: 98.1%
  sidewalk: 84.2%
  building: 92.1%
  ...
```

#### CamVid Test Set

```bash
python tools/eval.py \
  --cfg configs/camvid/pidnet_medium_camvid.yaml \
  TEST.MODEL_FILE pretrained_models/camvid/PIDNet_M_Camvid_Test.pt \
  DATASET.TEST_SET list/camvid/test.lst
```

### Step 3: Generate Test Predictions

For Cityscapes test set submission:

```bash
python tools/eval.py \
  --cfg configs/cityscapes/pidnet_large_cityscapes_trainval.yaml \
  TEST.MODEL_FILE pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt \
  DATASET.TEST_SET list/cityscapes/test.lst
```

Predictions are saved to: `output/cityscapes/pidnet_large/test_results/`

### Step 4: Understanding Evaluation Metrics

**mIoU (mean Intersection over Union):**
```
IoU = (Predicted ∩ Ground Truth) / (Predicted ∪ Ground Truth)
mIoU = Average of IoU across all classes
```

**Pixel Accuracy:**
```
Accuracy = Correct Pixels / Total Pixels
```

**Class-wise Performance:**
Each semantic class (e.g., road, car, person) has its own IoU score.

---

## 7. Running Inference on Custom Images

### Step 1: Prepare Your Images

```bash
# Place your images in the samples/ directory
mkdir -p samples
cp /path/to/your/images/*.png samples/
# OR *.jpg files
```

### Step 2: Run Inference

#### Using PIDNet-L (highest accuracy)

```bash
python tools/custom.py \
  --a 'pidnet-l' \
  --p 'pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt' \
  --t '.png'
```

#### Using PIDNet-S (fastest)

```bash
python tools/custom.py \
  --a 'pidnet-s' \
  --p 'pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt' \
  --t '.jpg'
```

#### Command-line Arguments:

- `--a`: Model architecture ('pidnet-s', 'pidnet-m', or 'pidnet-l')
- `--p`: Path to pretrained model
- `--t`: Image file extension ('.png', '.jpg', etc.)
- `--r`: Root directory for images (default: '../samples/')
- `--c`: Use Cityscapes classes (True) or CamVid classes (False)

### Step 3: View Results

Segmentation results are saved to: `samples/outputs/`

Each output image shows:
- Original image overlaid with segmentation mask
- Color-coded semantic classes (see color map in tools/custom.py)

### Step 4: Batch Processing

Process multiple image directories:

```bash
# Process all PNG images in custom_data/
python tools/custom.py \
  --a 'pidnet-m' \
  --p 'pretrained_models/cityscapes/PIDNet_M_Cityscapes_val.pt' \
  --r 'custom_data/' \
  --t '.png'
```

### Color Map for Cityscapes Classes

```python
0:  road (128, 64, 128)
1:  sidewalk (244, 35, 232)
2:  building (70, 70, 70)
3:  wall (102, 102, 156)
4:  fence (190, 153, 153)
5:  pole (153, 153, 153)
6:  traffic light (250, 170, 30)
7:  traffic sign (220, 220, 0)
8:  vegetation (107, 142, 35)
9:  terrain (152, 251, 152)
10: sky (70, 130, 180)
11: person (220, 20, 60)
12: rider (255, 0, 0)
13: car (0, 0, 142)
14: truck (0, 0, 70)
15: bus (0, 60, 100)
16: train (0, 80, 100)
17: motorcycle (0, 0, 230)
18: bicycle (119, 11, 32)
```

---

## 8. Understanding the Code Structure

### Repository Organization

```
PIDNet/
├── configs/              # Configuration files
│   ├── cityscapes/       # Cityscapes configs
│   ├── camvid/           # CamVid configs
│   └── default.py        # Default config
├── data/                 # Dataset root
│   ├── cityscapes/       # Cityscapes data
│   ├── camvid/           # CamVid data
│   └── list/             # Data split lists
├── datasets/             # Dataset loaders
│   ├── cityscapes.py     # Cityscapes dataset
│   ├── camvid.py         # CamVid dataset
│   └── base_dataset.py   # Base dataset class
├── models/               # Model implementations
│   ├── pidnet.py         # Main PIDNet model
│   ├── model_utils.py    # Building blocks
│   └── speed/            # Speed measurement
├── tools/                # Training/evaluation scripts
│   ├── train.py          # Training script
│   ├── eval.py           # Evaluation script
│   └── custom.py         # Custom inference
├── utils/                # Utility functions
│   ├── function.py       # Train/test functions
│   ├── criterion.py      # Loss functions
│   └── utils.py          # Helper utilities
└── pretrained_models/    # Pretrained weights
    ├── imagenet/         # ImageNet pretrained
    ├── cityscapes/       # Cityscapes finetuned
    └── camvid/           # CamVid finetuned
```

### Key Files Explained

#### 1. models/pidnet.py

The core model implementation:

```python
class PIDNet(nn.Module):
    def __init__(self, m, n, num_classes, planes, ...):
        # m: number of residual blocks in early layers
        # n: number of residual blocks in later layers
        # planes: base number of channels
        
        # I-Branch: Context extraction
        self.layer1 = self._make_layer(...)
        self.layer2 = self._make_layer(...)
        
        # P-Branch: Detail preservation
        self.layer3_ = self._make_layer(...)
        self.pag3 = PagFM(...)  # Fusion module
        
        # D-Branch: Boundary detection
        self.layer3_d = self._make_single_layer(...)
        self.diff3 = nn.Sequential(...)
```

#### 2. models/model_utils.py

Building blocks:
- `BasicBlock`: Standard residual block
- `Bottleneck`: Bottleneck residual block
- `PagFM`: Pixel-attention-guided fusion
- `PAPPM/DAPPM`: Pyramid pooling modules
- `Bag/Light_Bag`: Bilateral aggregation

#### 3. tools/train.py

Training pipeline:
1. Load configuration
2. Create model
3. Load pretrained weights
4. Setup optimizer and loss
5. Training loop with validation

#### 4. tools/eval.py

Evaluation pipeline:
1. Load model and weights
2. Run inference on test set
3. Calculate metrics (mIoU, accuracy)
4. Save predictions

#### 5. utils/criterion.py

Loss functions:
- `CrossEntropy`: Standard cross-entropy loss
- `OhemCrossEntropy`: Online hard example mining
- `BoundaryLoss`: Boundary-aware loss for D-branch

### Data Flow Through the Network

```
Input (3×1024×2048)
    ↓
conv1 (3→64, stride=4)
    ↓
layer1 (64→64)
    ↓
layer2 (64→128, stride=2)
    ↓
┌───────────────────┼───────────────────┐
│                   │                   │
│ P-Branch          │ I-Branch          │ D-Branch
│ (Detail)          │ (Context)         │ (Boundary)
│                   │                   │
│ layer3_           │ layer3            │ layer3_d
│ (128→128)         │ (128→256)         │ (128→64)
│   ↓               │   ↓               │   ↓
│ PagFM ←── compression3 ←─┘            │ diff3 ←────┘
│   ↓               │                   │   ↓
│ layer4_           │ layer4            │ layer4_d
│ (128→128)         │ (256→512)         │ (64→64)
│   ↓               │   ↓               │   ↓
│ PagFM ←── compression4 ←─┘            │ diff4 ←────┘
│   ↓               │                   │   ↓
│ layer5_           │ layer5→SPP        │ layer5_d
│ (128→256)         │ (512→1024)        │ (64→128)
│                   │                   │
└───────────────────┼───────────────────┘
                    ↓
            Bag/Light_Bag (Fusion)
                    ↓
            segmenthead (Final prediction)
                    ↓
            Output (19×128×256)
                    ↓
        Upsample to (19×1024×2048)
```

---

## 9. Advanced Topics

### 9.1 Measuring Inference Speed

PIDNet provides speed measurement scripts:

```bash
# Measure PIDNet-S on Cityscapes resolution
python models/speed/pidnet_speed.py \
  --a 'pidnet-s' \
  --c 19 \
  --r 1024 2048

# Measure PIDNet-M on CamVid resolution
python models/speed/pidnet_speed.py \
  --a 'pidnet-m' \
  --c 11 \
  --r 720 960
```

Output:
```
Model: pidnet-s
Input size: (1, 3, 1024, 2048)
Warmup iterations: 50
Speed test iterations: 100
Average FPS: 93.2
Average inference time: 10.7 ms
```

### 9.2 Fine-tuning on Custom Dataset

#### Step 1: Prepare Custom Dataset

Create a new dataset class in `datasets/custom_dataset.py`:

```python
from .base_dataset import BaseDataset

class CustomDataset(BaseDataset):
    def __init__(self, root, list_path, num_classes, ...):
        super(CustomDataset, self).__init__(...)
        self.num_classes = num_classes
        # Load your data
```

#### Step 2: Create Configuration

Create `configs/custom/pidnet_small_custom.yaml`:

```yaml
DATASET:
  DATASET: custom
  ROOT: data/custom/
  NUM_CLASSES: 10  # Your number of classes
  TRAIN_SET: 'list/custom/train.lst'
  TEST_SET: 'list/custom/val.lst'

MODEL:
  NAME: pidnet_small
  PRETRAINED: "pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt"

TRAIN:
  END_EPOCH: 200
  LR: 0.001  # Lower LR for fine-tuning
```

#### Step 3: Register Dataset

In `datasets/__init__.py`, add:

```python
from .custom_dataset import CustomDataset

__all__ = ['BaseDataset', 'cityscapes', 'CustomDataset']
```

#### Step 4: Train

```bash
python tools/train.py --cfg configs/custom/pidnet_small_custom.yaml
```

### 9.3 Multi-scale Testing

Enable multi-scale testing for higher accuracy:

```yaml
TEST:
  MULTI_SCALE: true
  FLIP_TEST: true
  SCALE_LIST: [0.5, 0.75, 1.0, 1.25, 1.5]
```

This tests the image at multiple scales and averages predictions.

### 9.4 Export to ONNX

Export model for deployment:

```python
import torch
import models

# Load model
model = models.pidnet.get_pred_model('pidnet-s', 19)
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 1024, 2048)
torch.onnx.export(
    model,
    dummy_input,
    "pidnet_s.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)
```

### 9.5 Optimize for Deployment

#### TensorRT Acceleration

Use TensorRT for faster inference:
1. Convert ONNX model to TensorRT engine
2. Expected speedup: 1.5-2x over PyTorch

#### Model Quantization

Reduce model size and increase speed:
```python
# Post-training quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Conv2d}, dtype=torch.qint8
)
```

### 9.6 Understanding Loss Functions

#### Multi-task Learning Losses

During training with `augment=True`, three losses are computed:

1. **Main Segmentation Loss** (P-branch output)
   ```python
   loss_p = criterion(pred_p, target)
   ```

2. **Auxiliary Segmentation Loss** (Final output)
   ```python
   loss_final = criterion(pred_final, target)
   ```

3. **Boundary Loss** (D-branch output)
   ```python
   loss_d = boundary_loss(pred_d, boundary_target)
   ```

Total loss:
```python
loss = 0.4 * loss_p + 1.0 * loss_final + 1.0 * loss_d
```

#### OHEM (Online Hard Example Mining)

Focuses on hard-to-classify pixels:
```yaml
LOSS:
  USE_OHEM: true
  OHEMTHRES: 0.9      # Keep pixels with loss > 0.9
  OHEMKEEP: 131072    # Keep at least 131072 pixels
```

---

## 10. Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**
1. Reduce batch size:
   ```bash
   TRAIN.BATCH_SIZE_PER_GPU 2
   ```

2. Use smaller model variant (PIDNet-S instead of PIDNet-L)

3. Reduce image size in config:
   ```yaml
   TRAIN:
     IMAGE_SIZE: [512, 512]  # Instead of [1024, 1024]
   ```

4. Enable gradient checkpointing (modify code):
   ```python
   from torch.utils.checkpoint import checkpoint
   ```

#### Issue 2: Dataset Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/cityscapes/...'
```

**Solutions:**
1. Check dataset path in config file
2. Verify data directory structure matches expected format
3. Update paths in data list files (`data/list/`)

#### Issue 3: Pretrained Model Loading Error

**Error:**
```
RuntimeError: Error(s) in loading state_dict for PIDNet
```

**Solutions:**
1. Verify model variant matches pretrained weights
2. Check if using correct pretrained file path
3. Try loading without strict mode:
   ```python
   model.load_state_dict(state_dict, strict=False)
   ```

#### Issue 4: Low Accuracy Results

**Possible Causes:**
1. Insufficient training epochs
2. Learning rate too high or too low
3. Data augmentation issues
4. Incorrect normalization

**Solutions:**
1. Train for more epochs (484 for Cityscapes)
2. Adjust learning rate:
   ```bash
   TRAIN.LR 0.005
   ```
3. Enable data augmentation:
   ```yaml
   TRAIN:
     FLIP: true
     MULTI_SCALE: true
   ```

#### Issue 5: Slow Training Speed

**Solutions:**
1. Use multiple GPUs:
   ```bash
   GPUS (0,1,2,3)
   ```

2. Increase number of workers:
   ```yaml
   WORKERS: 8
   ```

3. Enable cuDNN benchmark:
   ```yaml
   CUDNN:
     BENCHMARK: true
   ```

4. Use mixed precision training (requires apex or PyTorch 1.6+):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   ```

#### Issue 6: Visualization Issues

**Problem:** Output images look incorrect

**Solutions:**
1. Check color map matches dataset classes
2. Verify image preprocessing (mean/std normalization)
3. Ensure output is properly post-processed:
   ```python
   output = F.softmax(output, dim=1)
   pred = output.argmax(dim=1)
   ```

### Performance Optimization Tips

1. **Use DataLoader efficiently:**
   ```yaml
   WORKERS: 8  # Increase based on CPU cores
   TRAIN.SHUFFLE: true
   ```

2. **Pin memory for faster GPU transfer:**
   ```python
   train_loader = DataLoader(..., pin_memory=True)
   ```

3. **Use appropriate image size:**
   - Training: 1024×1024 (crop from 2048×1024)
   - Testing: 2048×1024 (full resolution)

4. **Monitor GPU utilization:**
   ```bash
   watch -n 1 nvidia-smi
   ```

### Getting Help

1. **Check the paper:** [arXiv:2206.02066](https://arxiv.org/abs/2206.02066)
2. **Official repository:** [GitHub Issues](https://github.com/XuJiacong/PIDNet/issues)
3. **Papers with Code:** [PIDNet Discussion](https://paperswithcode.com/paper/pidnet-a-real-time-semantic-segmentation)

---

## Summary and Next Steps

Congratulations! You've learned:

✅ **PIDNet architecture** - Understanding P, I, D branches  
✅ **Installation** - Setting up the environment  
✅ **Data preparation** - Organizing Cityscapes and CamVid datasets  
✅ **Training** - Training models from scratch  
✅ **Evaluation** - Measuring model performance  
✅ **Inference** - Running segmentation on custom images  
✅ **Advanced topics** - Speed measurement, fine-tuning, deployment  

### Recommended Learning Path

**Beginner:**
1. Start with inference on pretrained models (Section 7)
2. Understand the architecture (Section 2)
3. Try evaluation on standard datasets (Section 6)

**Intermediate:**
1. Train PIDNet-S on Cityscapes (Section 5)
2. Experiment with hyperparameters
3. Measure inference speed (Section 9.1)

**Advanced:**
1. Fine-tune on custom datasets (Section 9.2)
2. Implement multi-scale testing (Section 9.3)
3. Deploy with TensorRT (Section 9.5)

### Further Reading

- **Original Paper:** "PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller"
- **Related Work:** HRNet, DDRNet, BiSeNet for semantic segmentation
- **Control Theory:** Understanding PID controllers for better intuition

### Contributing

If you find issues or want to contribute:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

**Happy Learning! 🚀**

For questions or feedback, please open an issue on GitHub or refer to the official documentation.
