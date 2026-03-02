# Diagrams for Chapter 21: Model Selection and Cross-Validation

## Generated Diagrams

### 1. cross_validation_comparison.png
**Location:** Line 137 in content.md
**Purpose:** Visual comparison of single train/test split vs 5-fold cross-validation
**Type:** Conceptual diagram using matplotlib patches
**Dimensions:** 1400x400 pixels @ 150 DPI
**Color Scheme:**
- Training data: Blue (#2196F3)
- Test/Validation data: Red (#F44336)

**Description:** Shows how a single split uses data once (left panel) versus how 5-fold CV rotates through all folds ensuring every data point is validated exactly once (right panel).

---

### 2. single_split_vs_cv.png
**Location:** Generated in code example (line 327), used in walkthrough section
**Purpose:** Empirical demonstration of variance reduction via cross-validation
**Type:** Statistical comparison using histogram and bar chart
**Dimensions:** 1400x500 pixels @ 150 DPI
**Color Scheme:**
- Single split histogram: Blue (#2196F3)
- CV bar chart: Green (#4CAF50)
- Mean lines and confidence intervals: Red (#F44336)

**Description:** Left panel shows distribution of 20 random train/test splits demonstrating high variance. Right panel shows 5-fold CV scores with error bars, demonstrating lower variance and more reliable estimates.

---

## Generation Scripts

Both diagrams can be regenerated using:
```bash
python3 generate_diagram1.py
python3 generate_diagram2.py
```

## Color Palette Reference

Consistent colors used across all diagrams:
- Blue (#2196F3): Training data, primary comparisons
- Green (#4CAF50): Positive results, CV scores
- Orange (#FF9800): Warnings, intermediate states
- Red (#F44336): Test/validation data, mean lines
- Purple (#9C27B0): Advanced concepts
- Gray (#607D8B): Neutral elements

## Notes

- All diagrams use white backgrounds for print compatibility
- Font size minimum is 12pt for readability
- All figures saved at 150 DPI
- Maximum width is 800px as per style guide
- `plt.tight_layout()` called before saving for proper spacing
