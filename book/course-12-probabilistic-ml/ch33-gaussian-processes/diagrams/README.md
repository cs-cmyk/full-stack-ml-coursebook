# Chapter 33: Gaussian Processes - Diagrams

This directory contains all visualizations for Chapter 33 on Gaussian Processes.

## Generated Diagrams

### 1. gp_prior_samples.png
- **Type**: matplotlib
- **Description**: Sample functions from GP prior with RBF kernel
- **Shows**: 8 smooth, diverse function samples demonstrating uncertainty before observing data
- **Size**: 12x5 inches, 150 DPI

### 2. gp_posterior_update.png
- **Type**: matplotlib
- **Description**: GP posterior updates as observations arrive
- **Shows**: 4 panels showing evolution from prior to posterior with 0, 1, 3, and 8 observations
- **Size**: 14x10 inches, 150 DPI

### 3. lengthscale_comparison.png
- **Type**: matplotlib
- **Description**: Effect of lengthscale on GP function samples
- **Shows**: 3 panels comparing lengthscales 0.2, 1.0, and 3.0
- **Size**: 16x4 inches, 150 DPI

### 4. gp_regression_from_scratch.png
- **Type**: matplotlib
- **Description**: GP regression implementation from scratch on California Housing data
- **Shows**: Training data, GP mean prediction, and 95% confidence intervals
- **Size**: 12x6 inches, 150 DPI

### 5. gp_sklearn.png
- **Type**: matplotlib
- **Description**: GP regression using scikit-learn with optimized hyperparameters
- **Shows**: Same task as #4 but with automatic hyperparameter optimization
- **Size**: 12x6 inches, 150 DPI

### 6. kernel_composition_timeseries.png
- **Type**: matplotlib
- **Description**: Kernel composition for time series with trend and seasonality
- **Shows**: 3 panels comparing RBF, Periodic, and combined kernels
- **Size**: 14x12 inches, 150 DPI

### 7. hyperparameter_optimization.png
- **Type**: matplotlib
- **Description**: Hyperparameter optimization via marginal likelihood
- **Shows**: Log marginal likelihood curve and prediction comparison
- **Size**: 12x5 inches, 150 DPI

### 8. gp_classification.png
- **Type**: matplotlib
- **Description**: GP classification on breast cancer dataset
- **Shows**: Decision boundary with probabilities and uncertainty map (entropy)
- **Size**: 12x5 inches, 150 DPI

### 9. bayesian_optimization.png
- **Type**: matplotlib
- **Description**: Bayesian optimization process over 10 iterations
- **Shows**: 6 selected iterations showing GP posterior and sampling strategy
- **Size**: 16x14 inches, 150 DPI

## Color Palette

All diagrams use a consistent color scheme:
- Blue (#2196F3): Primary predictions, GP mean
- Green (#4CAF50): Next samples, positive indicators
- Orange (#FF9800): Alternative curves
- Red (#F44336): Observations, important markers
- Purple (#9C27B0): Additional curves
- Gray (#607D8B): Training data, background elements

## Technical Details

- All plots use white backgrounds for print compatibility
- Font size minimum: 12pt for axis labels
- All figures saved at 150 DPI
- Maximum width: 800px (16 inches @ 150 DPI)
- All plots use `plt.tight_layout()` for proper spacing
- Grid alpha set to 0.3 for subtle guidance lines

## Regenerating Diagrams

To regenerate all diagrams, run:
```bash
python generate_diagrams.py
```

This will overwrite all existing PNG files in this directory.

## Dependencies

- numpy
- matplotlib
- scipy
- scikit-learn

All diagrams are generated from the `generate_diagrams.py` script, which ensures reproducibility and consistency across all visualizations.
