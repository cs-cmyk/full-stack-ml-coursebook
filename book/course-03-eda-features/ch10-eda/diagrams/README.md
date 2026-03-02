# Chapter 10 EDA Diagrams

This directory contains all diagrams for Chapter 10: Exploratory Data Analysis.

## Diagram Files

### 1. The 5-Step EDA Framework
**File:** `01_eda_framework.mmd` (Mermaid)
**Type:** Flowchart
**Purpose:** Shows the systematic 5-step process for conducting EDA, including the iterative nature where discoveries loop back to earlier steps.
**Colors:** Blue (start), Green (steps), Orange (documentation), Purple (actions)

**Key Features:**
- Five main steps clearly labeled
- Arrows showing flow and iteration
- Documentation node emphasizing record-keeping
- Final action node showing modeling decisions

---

### 2. EDA Question Tree
**File:** `02_question_tree.mmd` (Mermaid)
**Type:** Mind map / Decision tree
**Purpose:** Illustrates the key questions to ask during EDA and the actions that follow from discoveries.
**Colors:** Blue (root), Green (questions), Orange (discoveries), Purple (actions)

**Key Features:**
- Central "Your Dataset" node
- Five main question branches
- Each branch leads to specific discoveries
- Action boxes showing what to do with findings

---

### 3. Visualization Cheat Sheet
**File:** `03_visualization_cheatsheet.png` (PNG, 150 DPI)
**Generator:** `create_viz_cheatsheet.py`
**Type:** 3×3 grid of example plots
**Purpose:** Quick reference guide showing 9 common EDA visualization types with use cases.

**Layout:**
- **Row 1 - Univariate:** Histogram, Box Plot, Count Plot
- **Row 2 - Bivariate:** Scatter Plot, Box by Group, Heatmap
- **Row 3 - Multivariate:** Pair Plot, Faceted Plot, Correlation Matrix

**Key Features:**
- Each plot has a clear example
- Use case labels ("Distribution shape", "Feature correlation", etc.)
- When-to-use guidance
- Consistent color palette throughout
- Professional annotations

---

### 4. From EDA Findings to Modeling Decisions
**File:** `04_findings_to_decisions.mmd` (Mermaid)
**Type:** Flow diagram
**Purpose:** Shows how common EDA findings translate into specific preprocessing decisions and modeling impacts.

**Structure:**
- **Left:** 5 common EDA findings (missing values, skewness, outliers, correlation, imbalance)
- **Middle:** Decision options for each finding
- **Right:** Positive modeling impacts

**Key Features:**
- Clear cause-and-effect relationships
- Multiple decision options per finding
- Shows why EDA matters for modeling
- Colors: Orange (findings), Blue (decisions), Green (impacts)

---

## Color Palette

All diagrams use the standardized color palette:
- **Blue (#2196F3):** Primary actions, decisions
- **Green (#4CAF50):** Steps, positive outcomes
- **Orange (#FF9800):** Warnings, discoveries, documentation
- **Red (#F44336):** Errors, negative classes
- **Purple (#9C27B0):** Final outcomes, advanced concepts
- **Gray (#607D8B):** Neutral elements

---

## Usage in content.md

See `DIAGRAM_INSERTION_GUIDE.md` for detailed instructions on where to insert each diagram into the chapter content.

**Quick reference:**
1. Diagram 1 → Section "The 5-Step EDA Framework" (~line 60)
2. Diagram 2 → After "The Detective Framework" (~line 40)
3. Diagram 3 → Before "Code Example 2" (~line 420)
4. Diagram 4 → Before "Common Pitfalls" (~line 928)

---

## Regenerating Diagrams

### Mermaid Diagrams (1, 2, 4)
These are source files that will be rendered by the markdown processor. No regeneration needed unless content changes.

### PNG Diagram (3)
To regenerate the visualization cheat sheet:

```bash
cd diagrams/
python create_viz_cheatsheet.py
```

This will create `03_visualization_cheatsheet.png` at 150 DPI, approximately 800x700 pixels.

---

## Design Principles

All diagrams follow these principles:
- ✅ Clear, readable labels (minimum 12pt font)
- ✅ Consistent color coding
- ✅ Educational focus (not just decorative)
- ✅ White backgrounds for print compatibility
- ✅ Appropriate resolution (150 DPI for images)
- ✅ Accessibility (color + text labels, not color alone)
- ✅ Self-contained (understandable without reading full text)

---

**Created:** 2026-02-28
**Author:** Diagram Agent
**Status:** Ready for integration into content.md
