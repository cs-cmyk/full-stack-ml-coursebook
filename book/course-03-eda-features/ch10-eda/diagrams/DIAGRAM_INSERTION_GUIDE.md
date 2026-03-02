# Diagram Insertion Guide for Chapter 10: EDA

This guide shows where to insert the generated diagrams into content.md.

## Generated Diagrams

1. ✅ `01_eda_framework.mmd` - The 5-Step EDA Framework (Mermaid flowchart)
2. ✅ `02_question_tree.mmd` - EDA Question Tree (Mermaid mind map)
3. ✅ `03_visualization_cheatsheet.png` - Visualization Cheat Sheet (Matplotlib figure)
4. ✅ `04_findings_to_decisions.mmd` - From EDA Findings to Modeling Decisions (Mermaid flowchart)

## Insertion Points in content.md

### 1. Insert Diagram 1: The 5-Step EDA Framework

**Location:** After line 58 in content.md (section header "## The 5-Step EDA Framework")

**Replace this text block (lines 60-86):**
```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   1. FIRST LOOK │────▶│ 2. INDIVIDUAL       │────▶│ 3. TARGET           │
│                 │     │    FEATURES         │     │    VARIABLE         │
│ • Shape         │     │                     │     │                     │
│ • Data types    │     │ • Distributions     │     │ • Class balance     │
│ • Memory usage  │     │ • Missing values    │     │   (classification)  │
│ • First rows    │     │ • Unique counts     │     │ • Distribution      │
│                 │     │ • Summary stats     │     │   (regression)      │
└─────────────────┘     └─────────────────────┘     └─────────────────────┘
         │                                                      │
         │                                                      │
         │              ┌─────────────────────┐                │
         └──────────────▶  ITERATE & DOCUMENT ◀────────────────┘
                        └─────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────┐            ┌─────────────────────┐
        │ 4. RELATIONSHIPS    │            │ 5. DATA QUALITY     │
        │                     │            │                     │
        │ • Feature-target    │            │ • Outliers          │
        │   correlations      │            │ • Duplicates        │
        │ • Feature-feature   │            │ • Inconsistencies   │
        │   interactions      │            │ • Logical errors    │
        │ • Patterns          │            │                     │
        └─────────────────────┘            └─────────────────────┘
```

**With this:**
```markdown
![The 5-Step EDA Framework](diagrams/01_eda_framework.mmd)

This framework provides a systematic approach where each step reveals information that guides the next steps. Discoveries often loop you back to re-examine earlier aspects with new questions—EDA is iterative, not linear. The key is to document your findings at each stage so you can make informed modeling decisions later.

**The framework in detail:**

1. **First Look** - Understand the dataset's structure: shape, data types, memory usage, and first rows
2. **Individual Features** - Explore each feature's distribution, missing values, unique counts, and summary statistics
3. **Target Variable** - Analyze class balance (classification) or distribution (regression)
4. **Relationships** - Discover feature-target correlations, feature-feature interactions, and patterns
5. **Data Quality** - Identify outliers, duplicates, inconsistencies, and logical errors
```

---

### 2. Insert Diagram 2: EDA Question Tree

**Location:** After line 39 (end of "The Detective Framework" section)

**Add this text:**
```markdown

As you investigate your dataset, keep these key questions in mind:

![EDA Question Tree: Key Questions to Ask](diagrams/02_question_tree.mmd)

Great detectives don't jump to conclusions—they examine evidence methodically, document findings carefully, and let the data reveal its secrets. That's exactly what we do in EDA, and this question tree guides our investigation at every step.
```

---

### 3. Insert Diagram 3: Visualization Cheat Sheet

**Location:** After line 419 (after the "Modeling Implications" section, before "## Code Example 2")

**Add this as a new subsection:**
```markdown

### Choosing the Right Visualization

Throughout EDA, visualization is one of your most powerful tools. Different plot types reveal different aspects of your data. Here's a quick reference guide:

![EDA Visualization Cheat Sheet](diagrams/03_visualization_cheatsheet.png)

**Quick selection guide:**
- **Univariate analysis** (one variable): Use histograms for distributions, box plots for outliers, count plots for categories
- **Bivariate analysis** (two variables): Use scatter plots for numeric relationships, box-by-group for comparing distributions, heatmaps for correlations
- **Multivariate analysis** (many variables): Use pair plots for all pairwise relationships, faceted plots to split by groups, correlation matrices for feature dependencies

The key is matching your visualization to your question: "What's the distribution?" calls for histograms, "How do features correlate?" calls for scatter plots or heatmaps, "How do groups differ?" calls for box-by-group plots.
```

---

### 4. Insert Diagram 4: From EDA Findings to Modeling Decisions

**Location:** After line 927 (after the messy data example section), before "## Common Pitfalls"

**Add this as a new subsection:**
```markdown

## From EDA Insights to Action

The ultimate goal of EDA is not just understanding—it's making informed decisions that improve your models. Every finding should translate into a concrete action. Here's how common EDA discoveries map to preprocessing and modeling strategies:

![From EDA Findings to Modeling Decisions](diagrams/04_findings_to_decisions.mmd)

This diagram shows the direct path from what you discover in EDA to the decisions you make in preprocessing and modeling. For example:

- **Finding:** Age has 20% missing values → **Decision:** Impute with median by sex/class → **Impact:** Better data quality without losing rows
- **Finding:** Fare is highly right-skewed → **Decision:** Apply log transform → **Impact:** Improved model performance on linear algorithms
- **Finding:** Strong correlation between radius, perimeter, and area → **Decision:** Use PCA or regularization → **Impact:** Reduced overfitting
- **Finding:** 62% vs 38% class distribution → **Decision:** Stratified CV, use F1 score → **Impact:** Fair model evaluation

The key lesson: **EDA is not exploratory for its own sake—it's exploratory to inform action.** Every plot, every statistic, every discovery should help you make better modeling decisions.
```

---

## Summary of Changes

**Files created:**
- `diagrams/01_eda_framework.mmd` (Mermaid diagram)
- `diagrams/02_question_tree.mmd` (Mermaid diagram)
- `diagrams/03_visualization_cheatsheet.png` (PNG image, 800x700px, 150 DPI)
- `diagrams/04_findings_to_decisions.mmd` (Mermaid diagram)
- `diagrams/create_viz_cheatsheet.py` (Generator script for diagram 3)

**Insertions needed in content.md:**
1. Line ~60: Replace ASCII art with Diagram 1 reference
2. Line ~40: Add Diagram 2 after Detective Framework
3. Line ~420: Add new subsection with Diagram 3
4. Line ~928: Add new subsection with Diagram 4

**Color palette used (as specified):**
- Blue: #2196F3
- Green: #4CAF50
- Orange: #FF9800
- Red: #F44336
- Purple: #9C27B0
- Gray: #607D8B

All diagrams follow the textbook style guide with:
- Clear titles and labels
- Consistent color scheme
- Educational focus
- Minimum 12pt font sizes
- White backgrounds
- 150 DPI for raster images
