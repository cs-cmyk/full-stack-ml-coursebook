# Diagram Generation Summary - Chapter 56: Causal ML in Practice

## Date
March 1, 2026

## Diagrams Generated

### 1. Four Quadrants for Uplift Modeling
- **File**: `diagrams/uplift_four_quadrants.png`
- **Size**: 129 KB
- **Resolution**: 150 DPI
- **Dimensions**: 800px width (max)
- **Type**: Matplotlib visualization
- **Description**: Visualizes the four customer segments in uplift modeling:
  - **Persuadables** (green): Customers who respond positively only when treated - TARGET THIS GROUP
  - **Sure Things** (orange): Customers who respond regardless - WASTE BUDGET
  - **Lost Causes** (gray): Customers who won't respond either way - WASTE BUDGET
  - **Sleeping Dogs** (red): Customers with negative treatment effects - AVOID!
- **Color Palette**:
  - Green: #2ecc71
  - Orange: #f39c12
  - Gray: #95a5a6
  - Red: #e74c3c

### 2. Uplift Curve
- **File**: `diagrams/uplift_curve.png`
- **Size**: 111 KB
- **Resolution**: 150 DPI
- **Dimensions**: 800px width (max)
- **Type**: Matplotlib visualization
- **Description**: Compares T-Learner model performance vs. random targeting, showing cumulative incremental response as more customers are targeted. Includes annotation for optimal targeting threshold.
- **Color Palette**:
  - Model curve: #2ecc71 (green)
  - Random baseline: #95a5a6 (gray)
  - Annotation: Yellow background

## Technical Details

### Implementation
- Both diagrams generated using matplotlib
- White backgrounds for readability
- Font size minimum 12pt
- Clear axis labels, titles, and legends
- `plt.tight_layout()` applied before saving
- Saved at 150 DPI with `bbox_inches='tight'`

### Data Generation
- Uplift curve diagram uses simulated marketing campaign data
- 10,000 customers with heterogeneous treatment effects
- T-Learner model trained with Random Forest classifiers
- Demonstrates 57.5% improvement over random targeting at 30% threshold

## Content Structure

The chapter content includes:
1. **Visualization section** (lines 54-113): Contains the code to generate the four-quadrants diagram
2. **Part 2: Uplift Curve Evaluation** (lines 270-378): Contains the complete code to train T-Learner model and generate uplift curve

The educational approach embeds the diagram generation code directly in the content to teach readers how to create these visualizations themselves. The actual PNG files are generated separately and stored in the `diagrams/` directory for inclusion in the final textbook.

## Files Created
1. `diagrams/uplift_four_quadrants.png` - Four quadrants visualization
2. `diagrams/uplift_curve.png` - Uplift curve comparison
3. `generate_diagrams.py` - Automated diagram generation script

## Status
✅ All diagrams successfully generated and saved
✅ Consistent color palette applied
✅ Proper formatting and styling
✅ Educational quality maintained
