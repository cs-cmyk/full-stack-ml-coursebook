# Chapter 11 Diagrams

This directory contains all diagrams for Chapter 11: Data Quality and Cleaning.

## Generated Diagrams

### 1. missing_data_types.png
**Location in content.md:** Referenced at line 129 as "Figure 11.1"
**Description:** Three scatter plots demonstrating the three types of missing data mechanisms:
- **MCAR (Missing Completely at Random)**: Red X's scattered randomly across all ages and income levels
- **MAR (Missing at Random)**: Red X's clustered in the older age group (Age > 50)
- **MNAR (Missing Not at Random)**: Red X's clustered in the high-income region (Income > $60k)

**Purpose:** Helps students understand that different missing data patterns require different handling strategies.

**Color Palette:**
- Blue (#2196F3): Observed data points
- Red (#F44336): Missing data indicators
- Orange (#FF9800): MAR threshold line
- Purple (#9C27B0): MNAR threshold line

---

### 2. quality_issues_visualization.png
**Location in content.md:** Generated in code example at line 245
**Description:** Four-panel visualization showing comprehensive data quality issues:
1. **Top-left:** Bar chart of missing value counts per column
2. **Top-right:** Heatmap showing missing data patterns in first 100 rows
3. **Bottom-left:** Box plot showing outliers in MedInc with IQR bounds
4. **Bottom-right:** Histogram comparing original vs. contaminated distributions

**Purpose:** Demonstrates how to systematically visualize multiple data quality dimensions simultaneously.

**Color Palette:**
- Orange (#FF9800): Missing data bars
- Blue (#2196F3): Original distribution / box plots
- Red (#F44336): Contaminated distribution / outlier bounds
- Yellow-Red gradient: Missing data heatmap

---

### 3. imputation_comparison.png
**Location in content.md:** Generated in code example at line 337
**Description:** Bar chart comparing RMSE (Root Mean Squared Error) across three imputation strategies:
- Mean Imputation
- Median Imputation
- KNN Imputation (typically performs best)

**Purpose:** Shows students that different imputation methods have measurable impact on model performance, and that KNN imputation often outperforms simpler methods.

**Color Palette:**
- Green (#4CAF50): Best performing strategy
- Orange (#FF9800): Other strategies
- Green dashed line: Best performance reference

---

## How to Regenerate

To regenerate all diagrams:

```bash
cd /home/chirag/ds-book/book/course-03-eda-features/ch11-data-quality/diagrams
python generate_diagrams.py
```

The script will:
1. Load the California Housing dataset
2. Introduce artificial quality issues (missing values, duplicates, outliers)
3. Generate all three visualizations
4. Save them as PNG files at 150 DPI

## Design Guidelines Followed

✓ Consistent color palette across all diagrams
✓ White backgrounds for print compatibility
✓ 150 DPI resolution for clarity
✓ Maximum 800px wide (quality_issues_visualization is 14" wide = ~1400px for detail)
✓ Font sizes ≥12pt for readability
✓ Clear axis labels, titles, and legends
✓ Grid lines for easier reading
✓ Tight layout to eliminate wasted space
✓ Annotations and labels where needed

## Notes

- These diagrams are also generated as part of the code examples in content.md
- The code shows students the complete process of creating these visualizations
- Students can run the code themselves to reproduce the diagrams
- The standalone `generate_diagrams.py` script is for maintenance and updates
