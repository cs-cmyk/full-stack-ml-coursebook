# Chapter 16 Time Features - Diagrams

## Generated Diagrams

This directory contains all diagrams for Chapter 16: Time Features.

### 1. time_features_overview.png
**Purpose**: Illustrates core concepts in time feature engineering
**Type**: matplotlib (6 subplots)
**Dimensions**: 14x10 inches, 150 DPI
**File size**: ~401 KB

**Subplots**:
- A. Linear Encoding Problem - Shows why linear hour encoding creates artificial distance
- B. Cyclical Encoding (Circle) - Hours mapped to a circle with polar coordinates
- C. Sine-Cosine Transformation - Sine and cosine waves preserve circular distance
- D. Lag Features - Demonstrates shift operations for temporal dependencies
- E. Rolling Window Features - Shows smoothing effect of rolling averages
- F. Temporal Split - Correct vs incorrect train/test splitting

**Referenced in**: Line 164 of content.md (in code example)

### 2. time_features_predictions.png
**Purpose**: Shows model predictions and residuals for time series forecasting
**Type**: matplotlib (2 subplots)
**Dimensions**: 14x8 inches, 150 DPI
**File size**: ~329 KB

**Subplots**:
- Top: Actual vs Predicted Sales (last week of test set)
- Bottom: Residual Plot showing prediction errors

**Referenced in**: Lines 376, 379, 447 of content.md (in code example)

## Color Palette Used

Consistent with textbook standards:
- Blue (#2196F3) - Primary data series
- Green (#4CAF50) - Secondary series, positive indicators
- Orange (#FF9800) - Tertiary series, warnings
- Red (#F44336) - Errors, problems, negative indicators
- Purple (#9C27B0) - Additional data points
- Gray (#607D8B) - Background/less important data

## Generation Scripts

- `generate_overview.py` - Creates time_features_overview.png
- `generate_predictions.py` - Creates time_features_predictions.png

Both scripts can be re-run to regenerate diagrams if needed.

## Notes

- All diagrams use white backgrounds for print compatibility
- Font sizes are minimum 12pt for readability
- All figures include proper axis labels, titles, and legends
- Tight layout applied to prevent label cutoff
- Diagrams are integrated as code output examples in the content
