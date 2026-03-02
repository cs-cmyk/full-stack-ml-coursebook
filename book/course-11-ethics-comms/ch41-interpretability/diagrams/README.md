# Chapter 41: Model Interpretability & Explainability - Diagrams

## Generated Diagrams

All diagrams have been successfully generated and integrated into content.md.

### 1. Interpretability Spectrum
- **File**: `interpretability_spectrum.png`
- **Type**: Matplotlib horizontal bar chart
- **Purpose**: Shows the interpretability spectrum from glass-box models (linear regression) to black-box models (large transformers)
- **Location in content.md**: Line 85

### 2. Permutation Feature Importance
- **File**: `permutation_importance.png`
- **Type**: Matplotlib horizontal bar chart
- **Purpose**: Demonstrates permutation importance for Random Forest on breast cancer dataset
- **Location in content.md**: Line 235

### 3. SHAP Waterfall Plot
- **File**: `shap_waterfall.png`
- **Type**: SHAP waterfall visualization
- **Purpose**: Shows local explanation for a single prediction, displaying how features contribute to the final prediction
- **Location in content.md**: Line 305

### 4. SHAP Summary Plot
- **File**: `shap_summary.png`
- **Type**: SHAP beeswarm plot
- **Purpose**: Global explanation showing SHAP values for all test instances, revealing feature importance and effects
- **Location in content.md**: Line 347

### 5. Partial Dependence and ICE Plots
- **File**: `pdp_ice_plot.png`
- **Type**: Matplotlib PDP/ICE visualization
- **Purpose**: Shows both average effect (PDP) and individual effects (ICE) of median income on house prices
- **Location in content.md**: Line 512

### 6. Wine Feature Importance Comparison
- **File**: `wine_importance_comparison.png`
- **Type**: Side-by-side bar charts
- **Purpose**: Compares permutation importance vs SHAP importance for Wine dataset (Solution 1)
- **Location in content.md**: Line 720

### 7. Diabetes PDP and ICE Plots
- **File**: `diabetes_pdp_ice.png`
- **Type**: Matplotlib PDP/ICE visualization
- **Purpose**: Shows heterogeneity in BMI effects on diabetes progression (Solution 2)
- **Location in content.md**: Line 777

## Color Palette Used
- Blue (#2196F3): Primary color for bars and emphasis
- Orange (#FF9800): Secondary color for SHAP comparisons
- Red (#F44336): PDP average lines
- Gray (#607D8B): ICE individual lines
- Green (#4CAF50): Glass-box region indicators

## Technical Details
- All diagrams saved at 150 DPI for optimal quality
- Max width: 800px (enforced through figure sizing)
- White backgrounds for print compatibility
- Consistent font sizing (12pt minimum for readability)
- All diagrams use `tight_layout()` before saving

## Generation Scripts
Each diagram has a corresponding Python script in this directory:
- `generate_interpretability_spectrum.py`
- `generate_permutation_importance.py`
- `generate_shap_waterfall.py`
- `generate_shap_summary.py`
- `generate_pdp_ice.py`
- `generate_wine_importance.py`
- `generate_diabetes_pdp.py`

To regenerate all diagrams:
```bash
cd diagrams/
python generate_interpretability_spectrum.py
python generate_permutation_importance.py
python generate_shap_waterfall.py
python generate_shap_summary.py
python generate_pdp_ice.py
python generate_wine_importance.py
python generate_diabetes_pdp.py
```
