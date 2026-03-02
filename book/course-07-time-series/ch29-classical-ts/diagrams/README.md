# Chapter 29 Diagrams

This directory contains all visualizations for Chapter 29: Classical Time Series Analysis.

## Generated Diagrams

All diagrams were generated using the consistent color palette:
- Blue (#2196F3, #2E86AB): Primary data, training data
- Purple (#9C27B0, #A23B72): Trend, test data
- Orange (#FF9800): Seasonal patterns, forecasts
- Green (#4CAF50, #6A994E): Residuals, alternative models
- Red (#F44336): Highlights, warnings
- Gray (#607D8B): Secondary elements

### Main Diagrams

1. **components.png** (259 KB)
   - Figure 29.1 - Main time series decomposition visualization
   - Shows original series, trend, seasonal, and residual components
   - Used in the "Visualization" section

2. **raw_series.png** (92 KB)
   - Original Air Passengers time series (1949-1960)
   - Used in Part 1 code example

3. **decomposition.png** (246 KB)
   - Detailed STL decomposition plot
   - Used in Part 2 code example

4. **stationarity_process.png** (315 KB)
   - Four-panel visualization showing differencing transformations
   - Original, first-order diff, seasonal diff, and both
   - Used in Part 3 code example

5. **acf_pacf.png** (59 KB)
   - Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
   - Used in Part 4 code example

6. **sarima_forecast.png** (160 KB)
   - SARIMA(1,1,1)(1,1,1)₁₂ forecast with 95% confidence intervals
   - Used in Part 5 code example

7. **exponential_smoothing.png** (307 KB)
   - Four-panel comparison of exponential smoothing methods
   - SES, Holt's Linear, HW Additive, HW Multiplicative
   - Used in Part 6 code example

8. **final_comparison.png** (191 KB)
   - Side-by-side comparison of SARIMA vs Holt-Winters
   - Used in Part 7 code example

## Technical Details

- **Format**: PNG
- **Resolution**: 150 DPI
- **Background**: White
- **Maximum width**: 800px (as per guidelines)
- **Font sizes**: Minimum 12pt for readability
- **Grid**: Alpha 0.3 for subtle background
- **All plots**: Use `plt.tight_layout()` before saving

## Regeneration

To regenerate all diagrams:

```bash
cd book/course-07-time-series/ch29-classical-ts/diagrams
python generate_all.py
```

The script will:
1. Load the Air Passengers dataset
2. Perform all necessary transformations
3. Fit all models (SARIMA, exponential smoothing)
4. Generate all 8 diagrams
5. Save as PNG files at 150 DPI

## Data Source

All diagrams use the classic "Air Passengers" dataset from R's datasets package, accessed via statsmodels:
- Monthly totals of international airline passengers (1949-1960)
- 144 observations
- Strong trend and seasonal components
- Ideal for teaching time series decomposition and forecasting
