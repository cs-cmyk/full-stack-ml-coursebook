# Chapter 37 Diagrams

This directory contains all generated diagrams for Chapter 37: Computer Vision - Image Preprocessing and Augmentation.

## Generated Diagrams

### 1. image_tensor_structure.png
**Purpose**: Illustrates how images are represented as 3D tensors and shows the effect of different normalization techniques.

**Content**:
- Original sample image (32×32×3)
- Separated R, G, B channels
- Pixel value distributions at different normalization stages:
  - Original [0, 255]
  - Min-Max normalized [0, 1]
  - Standardized (mean≈0, std≈1)
- Tensor structure diagram

**Referenced in**: Section "Visualization" (line 30-115)

**Suggested placement**: After the formal definition section, to visually explain the concepts.

---

### 2. normalization_comparison.png
**Purpose**: Compares different normalization approaches side-by-side.

**Content**:
- Original image [0, 255]
- Min-Max normalized [0, 1]
- Standardized (mean≈0, std≈1)
- ImageNet normalized (for transfer learning)

**Referenced in**: Part 1 example code (line 199-233)

**Suggested placement**: In the "Loading and Normalizing Images" example section.

---

### 3. augmentation_variations.png
**Purpose**: Demonstrates how data augmentation creates diverse variations from a single image.

**Content**:
- 3×3 grid showing 9 different augmented versions of the same image
- Each applies random: flip, rotation, crop, and color jitter

**Referenced in**: Part 2 example code (line 306-327)

**Suggested placement**: In the "Basic Augmentation Pipeline" example section.

---

### 4. augmentation_performance_comparison.png
**Purpose**: Shows the impact of augmentation on model performance and overfitting.

**Content**:
- **Left plot**: Training vs test accuracy curves for three strategies
  - No augmentation (blue)
  - Basic augmentation (green)
  - Advanced augmentation (red)
- **Right plot**: Overfitting gap (train - test accuracy) over time
  - Demonstrates how augmentation reduces overfitting

**Referenced in**: Part 3 example code (line 601-637)

**Suggested placement**: In the "Advanced Augmentation Techniques and Impact on Performance" section.

---

## Integration Notes

The current content.md includes these diagrams as outputs of executable Python code examples. The diagrams have been pre-generated and are available in this directory for:

1. **Quick reference**: Readers can see the expected outputs without running the code
2. **Documentation**: The diagrams serve as visual aids in presentations or printed materials
3. **Testing**: Verify that example code produces expected visualizations

## Technical Details

- **Resolution**: 150 DPI (suitable for screen and print)
- **Maximum width**: 800px (as per guidelines)
- **Format**: PNG with white background
- **Color palette**: Consistent use of the standard palette:
  - Blue: #2196F3
  - Green: #4CAF50
  - Orange: #FF9800
  - Red: #F44336
  - Purple: #9C27B0
  - Gray: #607D8B

## Regeneration

To regenerate all diagrams, run:

```bash
python generate_diagrams_simple.py
```

This will overwrite existing diagrams with newly generated versions.
