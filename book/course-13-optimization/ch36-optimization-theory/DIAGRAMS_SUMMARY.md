# Chapter 36: Optimization Theory - Diagrams Summary

## Generated Diagrams

All diagrams for Chapter 36 have been successfully generated and referenced in content.md.

### 1. Convex vs Non-Convex Functions
- **File**: `diagrams/convex_vs_nonconvex.png`
- **Type**: Matplotlib 3D surface plots and 2D contour plot
- **Description**: Three-panel visualization showing:
  - Left: 3D surface of a convex paraboloid function
  - Center: 3D surface of a non-convex Himmelblau function with multiple local minima
  - Right: Contour plot with gradient descent paths from different starting points, all converging to the global minimum
- **Purpose**: Demonstrates the fundamental difference between convex and non-convex optimization landscapes
- **Referenced at**: Line 162

### 2. Lagrange Multipliers Geometric Interpretation
- **File**: `diagrams/lagrange_geometric.png`
- **Type**: Matplotlib 2D contour plot
- **Description**: Shows elliptical contours of the objective function f(x,y) = x² + 4y², a linear constraint x + 2y = 2, the optimal point, and gradient vectors demonstrating that ∇f = λ∇g at the optimum
- **Purpose**: Illustrates the geometric meaning of Lagrange multipliers - optimum occurs where objective contour is tangent to constraint
- **Referenced at**: Line 164

### 3. SVM Dual Formulation with KKT Conditions
- **File**: `diagrams/svm_dual.png`
- **Type**: Matplotlib (2 subplots)
- **Description**:
  - Left: SVM decision boundary with support vectors highlighted on Iris dataset
  - Right: Bar chart showing support vector indicators for all samples
- **Purpose**: Demonstrates SVM dual formulation, support vectors, and KKT conditions
- **Referenced at**: Line 487

### 4. Non-Negative Matrix Factorization
- **File**: `diagrams/nmf_projected_gd.png`
- **Type**: Matplotlib (3 rows of subplots)
- **Description**:
  - Top: Convergence curve showing reconstruction error over iterations
  - Middle: 10 learned basis vectors from digit images
  - Bottom: Original vs reconstructed digit images
- **Purpose**: Shows projected gradient descent for constrained optimization with non-negativity constraints
- **Referenced at**: Line 639

### 5. Gradient Descent vs L-BFGS
- **File**: `diagrams/gd_vs_lbfgs.png`
- **Type**: Matplotlib (2 subplots)
- **Description**:
  - Left: Loss vs iterations comparison
  - Right: Loss vs wall-clock time comparison
- **Purpose**: Compares first-order (gradient descent) and second-order (L-BFGS) optimization methods on logistic regression
- **Referenced at**: Line 831

### 6. Neural Network Loss Landscape
- **File**: `diagrams/loss_landscape.png`
- **Type**: Matplotlib (3 subplots including 3D)
- **Description**:
  - Left: 3D surface plot of loss landscape around optimal parameters
  - Center: Contour plot with training trajectory overlay
  - Right: Learned decision boundary on 2D dataset
- **Purpose**: Visualizes the non-convex nature of neural network optimization
- **Referenced at**: Line 1055

## Color Palette

All diagrams use a consistent color palette:
- Blue: `#2196F3`
- Green: `#4CAF50`
- Orange: `#FF9800`
- Red: `#F44336`
- Purple: `#9C27B0`
- Gray: `#607D8B`

## Technical Specifications

- **Resolution**: 150 DPI
- **Format**: PNG with white background
- **Font sizes**: Title 12pt, labels 11pt, ticks 9pt
- **Maximum width**: 800px (as per guidelines)

## Generation Script

The diagrams can be regenerated using:
```bash
cd book/course-13/ch36
python generate_diagrams.py
```

This script handles missing dependencies gracefully and provides fallback visualizations when needed (e.g., PyTorch for neural network loss landscape).

## Files Created

1. `generate_diagrams.py` - Main script to generate all diagrams
2. `add_image_refs.py` - Script to add image references to content.md
3. `diagrams/convex_vs_nonconvex.png` - 823 KB
4. `diagrams/lagrange_geometric.png` - 194 KB
5. `diagrams/svm_dual.png` - 145 KB
6. `diagrams/nmf_projected_gd.png` - 69 KB
7. `diagrams/gd_vs_lbfgs.png` - 68 KB
8. `diagrams/loss_landscape.png` - 577 KB

Total size: ~1.9 MB
