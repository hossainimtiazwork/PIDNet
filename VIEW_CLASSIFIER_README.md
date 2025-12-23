# View Classification Auxiliary Head for PIDNet

## Overview

This implementation adds an auxiliary view classification head to the PIDNet architecture. The view classifier can classify input images into N categories (e.g., different viewpoints, scene types, or other global image properties).

## Architecture Decision: Why the I (Integral) Branch?

PIDNet has three branches:
- **P (Proportional) Branch**: Focuses on detail preservation (local features)
- **I (Integral) Branch**: Focuses on context embedding (global semantic understanding)
- **D (Derivative) Branch**: Focuses on boundary detection (edge features)

**The view classifier is attached to the I (Integral) branch** because:

1. **Global Context**: View classification is a global task that requires understanding the overall scene context, not just local details or edges.

2. **Semantic Features**: The I branch processes features through the deepest layers (layer5) which contain the richest semantic information needed for scene-level classification.

3. **Feature Quality**: The I branch has access to multi-scale features and goes through the complete network depth, making it ideal for understanding scene-level properties.

4. **Architecture Design**: According to the PIDNet paper, the I branch is the main branch responsible for context embedding, which aligns perfectly with view classification requirements.

## Implementation Details

### Model Changes (`models/pidnet.py`)

1. **New Parameter**: `num_view_classes` - Number of view categories to classify (0 disables the feature)

2. **View Classifier Head**: 
   ```python
   nn.Sequential(
       nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
       nn.Flatten(),
       nn.Linear(planes * 16, planes * 4),  # Feature reduction
       nn.BatchNorm1d(planes * 4),
       nn.ReLU(inplace=True),
       nn.Dropout(0.5),
       nn.Linear(planes * 4, num_view_classes)  # Classification
   )
   ```

3. **Feature Extraction**: Uses features from the I branch after `layer5` (before SPP module)

4. **Output Format**: 
   - When `augment=True` and view classifier enabled: `[x_extra_p, x_, x_extra_d, x_view]`
   - When `augment=True` without view classifier: `[x_extra_p, x_, x_extra_d]`
   - When `augment=False` with view classifier: `x_, x_view`
   - When `augment=False` without view classifier: `x_`

### Loss Changes (`utils/criterion.py`)

1. **New Loss Class**: `ViewClassificationLoss`
   - Uses CrossEntropyLoss for multi-class classification
   - Supports optional class weights

### Training Changes

1. **FullModel Wrapper** (`utils/utils.py`):
   - Accepts optional `view_loss` parameter
   - Computes view classification loss when enabled
   - Combines all losses: `loss = loss_s + loss_b + loss_sb + view_weight * loss_view`

2. **Training Function** (`utils/function.py`):
   - Updated to handle optional `view_labels` parameter
   - Currently passes `None` for view_labels (requires dataset modification)

## Configuration

### Example Configuration (`configs/cityscapes/pidnet_small_cityscapes_view.yaml`)

```yaml
MODEL:
  NAME: pidnet_small
  NUM_OUTPUTS: 2
  NUM_VIEW_CLASSES: 4  # Number of view categories
  PRETRAINED: "pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar"

LOSS:
  USE_OHEM: true
  OHEMTHRES: 0.9
  OHEMKEEP: 131072
  BALANCE_WEIGHTS: [0.4, 1.0]
  SB_WEIGHTS: 1.0
  VIEW_WEIGHT: 0.5  # Weight for view classification loss (adjustable)
```

### Configuration Parameters

- **NUM_VIEW_CLASSES**: Number of view categories (set to 0 or omit to disable)
- **VIEW_WEIGHT**: Weight for view classification loss in total loss computation (default: 1.0)

## Usage

### Training with View Classifier

```bash
python tools/train.py --cfg configs/cityscapes/pidnet_small_cityscapes_view.yaml GPUS (0,1)
```

### Training without View Classifier (Original Behavior)

```bash
python tools/train.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml GPUS (0,1)
```

## Dataset Modifications Required

To use the view classifier in training, you need to modify your dataset class to provide view labels:

1. Add view labels to your dataset (e.g., in `datasets/cityscapes.py`)
2. Return view labels in `__getitem__`: `return image, label, edge, size, name, view_label`
3. Update training loop to extract and use view labels

Example modification for Cityscapes dataset:

```python
def __getitem__(self, index):
    item = self.files[index]
    name = item["name"]
    
    # ... existing code ...
    
    # Extract view label from filename or metadata
    view_label = self.get_view_label(name)  # Implement based on your needs
    
    return image.copy(), label.copy(), edge.copy(), np.array(size), name, view_label
```

## Testing and Inference

When loading a model trained with view classifier for inference:

```python
from models.pidnet import get_pred_model

# Load model with view classifier
model = get_pred_model(name='pidnet-s', num_classes=19, num_view_classes=4)
model.load_state_dict(torch.load('path/to/checkpoint.pt'))
model.eval()

# Forward pass
with torch.no_grad():
    if model.num_view_classes > 0:
        seg_output, view_pred = model(input_image)
    else:
        seg_output = model(input_image)
```

## Benefits

1. **Multi-task Learning**: Joint training of segmentation and view classification can improve both tasks through shared feature representations.

2. **Scene Understanding**: View classification provides additional context that can be useful for downstream tasks.

3. **Minimal Overhead**: The view classifier adds minimal computational cost (just global pooling and a few FC layers).

4. **Backward Compatible**: The feature is completely optional and doesn't affect existing functionality when disabled.

## Experiments and Tuning

### Recommended Hyperparameters

- **VIEW_WEIGHT**: Start with 0.5 and adjust based on validation performance
  - Increase if view classification accuracy is too low
  - Decrease if segmentation performance degrades

- **Dropout**: 0.5 is a good starting point for the view classifier head

### Loss Balancing

The total loss is computed as:
```
total_loss = loss_s + loss_b + loss_sb + VIEW_WEIGHT * loss_view
```

Where:
- `loss_s`: Semantic segmentation loss
- `loss_b`: Boundary loss  
- `loss_sb`: Semantic boundary loss
- `loss_view`: View classification loss

## Future Improvements

1. **Attention Mechanism**: Add attention to weight I-branch features based on view prediction
2. **Feature Fusion**: Use view predictions to modulate segmentation features
3. **Multi-scale View Features**: Extract view features from multiple layers
4. **Contrastive Learning**: Use view labels for contrastive pre-training

## References

- PIDNet Paper: "PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller" (CVPR 2023)
- Implementation based on: https://github.com/XuJiacong/PIDNet
