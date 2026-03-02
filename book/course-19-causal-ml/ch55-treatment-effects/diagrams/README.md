# Chapter 55 Diagrams

This directory contains all visualizations for Chapter 55: Average Treatment Effect (ATE).

## Generated Diagrams

### 1. ate_potential_outcomes.png
- **Type**: matplotlib (2-panel figure)
- **Purpose**: Illustrates the potential outcomes framework
- **Left panel**: Shows 8 individuals with their Y₀ (blue) and Y₁ (red) potential outcomes
  - Bold borders indicate observed outcomes
  - Faded dots show counterfactual (unobserved) outcomes
- **Right panel**: Bar chart showing E[Y₀] and E[Y₁] with ATE as the difference
- **Referenced in**: Line 129 of content.md (Visualization section)

### 2. ate_overlap_balance.png
- **Type**: matplotlib (2-panel figure)
- **Purpose**: Demonstrates propensity score overlap and covariate balance checks
- **Left panel**: Propensity score distributions for treated vs control groups
  - Shows common support region (between 0.1 and 0.9)
  - Overlapping histograms indicate good positivity
- **Right panel**: Standardized mean differences for key covariates
  - Shows imbalance before adjustment (SMD > 0.1 threshold)
  - Visualizes confounding in treatment assignment
- **Referenced in**: Line 257 of content.md (Part 2: Checking Overlap)

### 3. ate_method_comparison.png
- **Type**: matplotlib (horizontal bar chart)
- **Purpose**: Compares performance of different ATE estimation methods
- **Methods shown**:
  - Naive (Biased) - in red
  - Regression Adjustment - in blue
  - Inverse Propensity Weighting - in orange
  - Doubly Robust - in green
- **Features**: Shows bias for each method relative to true ATE (dashed line)
- **Referenced in**: Line 366 of content.md (Part 3: Estimating ATE)

## Color Palette
Consistent with textbook standards:
- **#2196F3**: Blue (control/Y₀)
- **#F44336**: Red (treated/Y₁)
- **#FF9800**: Orange (IPW method)
- **#4CAF50**: Green (doubly robust method)
- **#607D8B**: Gray (thresholds/reference lines)

## Technical Specifications
- **Resolution**: 150 DPI
- **Background**: White
- **Format**: PNG
- **Maximum width**: 800px (achieved through figsize settings)
- **Font size**: Minimum 12pt for readability
