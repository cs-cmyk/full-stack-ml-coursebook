# Chapter 7 Diagrams

This directory contains all visual assets for Chapter 7: Data Visualization with Matplotlib and Seaborn.

## Generated Diagrams

### Conceptual Diagrams

1. **matplotlib_architecture.svg**
   - Type: SVG (vector graphic)
   - Purpose: Illustrates the object-oriented hierarchy in Matplotlib (Figure → Axes → Axis → Artists)
   - Location in chapter: After "Matplotlib Architecture" section
   - Style: Clean, educational diagram with labeled components

2. **chart_decision_tree.mmd**
   - Type: Mermaid flowchart
   - Purpose: Decision tree for selecting appropriate chart types based on data and goals
   - Location in chapter: "Visual Overview" section
   - Note: Requires Mermaid rendering support

### Example Visualizations (California Housing Dataset)

All example visualizations use the California Housing dataset with consistent styling:
- DPI: 150
- Color palette: Colorblind-safe palette
- Style: White grid background
- Font sizes: 11-13pt for readability

3. **univariate_analysis.png**
   - 4-panel figure showing distribution analysis
   - Panels: Histogram (income), Box plot (income), Histogram (house value), KDE (house value)
   - Demonstrates: Distribution visualization techniques, outlier detection, skewness

4. **bivariate_analysis.png**
   - 2-panel figure showing relationships between variables
   - Panels: Scatter plot (with color encoding), Hexbin plot (density)
   - Demonstrates: Correlation visualization, handling overplotting

5. **categorical_comparison.png**
   - 3-panel figure comparing distributions across categories
   - Panels: Box plot, Violin plot, Strip plot with means
   - Demonstrates: Group comparison techniques, statistical summaries

6. **multivariate_analysis.png**
   - 2-panel figure showing multi-variable relationships
   - Panels: Correlation heatmap, Scatter with multiple encodings
   - Demonstrates: Correlation matrices, multi-dimensional data visualization

7. **publication_quality.png**
   - Single polished figure with professional styling
   - Features: 4 data dimensions (x, y, color, size), trend line, annotations
   - Demonstrates: Publication-ready figure creation

### Bonus Diagrams

8. **chart_types_comparison.png**
   - 6-panel overview of common chart types
   - Panels: Histogram, Box plot, Scatter plot, Line plot, Bar chart, Heatmap
   - Purpose: Quick reference for chart type selection
   - Location in chapter: After "Visual Overview" section

## Regenerating Diagrams

To regenerate all diagrams:

```bash
cd diagrams/
python3 generate_diagrams.py
```

This will:
1. Load the California Housing dataset
2. Generate all 6 visualization examples
3. Save them as PNG files (150 DPI, max 800px wide)
4. Display progress messages

## File Sizes

- SVG files: ~5KB (scalable)
- Mermaid files: ~2KB (text-based)
- PNG files: 150-650KB each (high quality for textbook printing)

## Color Palette

Consistent colors used across all diagrams:
- Blue: `#2196F3` (primary)
- Green: `#4CAF50` (success/positive)
- Orange: `#FF9800` (warning/highlight)
- Red: `#F44336` (error/important)
- Purple: `#9C27B0` (accent)
- Gray: `#607D8B` (neutral)

All colors are colorblind-accessible.

## Dependencies

Python packages required:
- numpy
- pandas
- matplotlib >= 3.5
- seaborn >= 0.12
- scikit-learn (for dataset)

Install with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
