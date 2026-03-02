# Chapter 8: SQL Databases - Diagrams

This directory contains all educational diagrams for Chapter 8: SQL Databases for Data Science.

## Generated Diagrams

### 1. csv_vs_database.png
**Purpose:** Visual comparison of CSV-based vs Database-based data workflows
**Type:** Matplotlib diagram
**Used in:** Visual section (beginning of chapter)
**Key concepts:**
- Side-by-side comparison of approaches
- Shows CSV limitations (memory, speed, multi-user)
- Demonstrates database advantages (filtering, indexing, concurrent access)

### 2. sql_execution_order.png
**Purpose:** Demonstrates SQL query execution order vs written order
**Type:** Matplotlib flowchart
**Used in:** Visual section (SQL Query Execution Order)
**Key concepts:**
- Written order: SELECT → FROM → WHERE → GROUP BY → HAVING → ORDER BY → LIMIT
- Execution order: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
- Includes warning about common errors (e.g., can't use SELECT alias in WHERE)

### 3. groupby_example.png
**Purpose:** Example bar chart of GROUP BY query results
**Type:** Matplotlib bar chart
**Used in:** Code Example section (Part 4)
**Key concepts:**
- Shows average house values by region
- Example of SQL aggregation visualization
- Demonstrates SQL → pandas → matplotlib workflow

### 4. where_vs_having.png
**Purpose:** Visual comparison of WHERE clause vs HAVING clause
**Type:** Matplotlib diagram
**Used in:** Common Pitfalls section (Pitfall #1)
**Key concepts:**
- WHERE filters rows BEFORE grouping
- HAVING filters groups AFTER aggregation
- Visual representation of when each executes in the pipeline

### 5. join_types.png
**Purpose:** Comprehensive guide to SQL JOIN types (INNER, LEFT, RIGHT, FULL OUTER)
**Type:** Matplotlib diagram with Venn-style visualizations
**Used in:** Appendix A (JOIN Types Visual Guide)
**Key concepts:**
- Four-panel layout showing each JOIN type
- Venn diagram style with overlapping circles
- Shows sample results for each JOIN type
- Includes "guests" and "meals" example tables

## Style Guide

All diagrams follow consistent styling:

### Colors
- **Blue (#2196F3):** Primary elements, database components, FROM clause
- **Green (#4CAF50):** Positive outcomes, successful operations, matched records
- **Orange (#FF9800):** Warnings, WHERE clause, intermediate steps
- **Red (#F44336):** Errors, NULL values, problematic approaches
- **Purple (#9C27B0):** HAVING clause, aggregate operations
- **Gray (#607D8B):** Secondary text, labels, written order

### Technical Specifications
- **Resolution:** 150 DPI
- **Max width:** ~800px (varies by diagram type)
- **Background:** White (#FFFFFF)
- **Minimum font size:** 12pt for all body text
- **Layout:** All diagrams use `plt.tight_layout()` for clean borders

### Typography
- **Titles:** 16-18pt, bold
- **Section headers:** 14pt, bold
- **Body text:** 11-12pt
- **Code/monospace:** 10-11pt, monospace family
- **Annotations:** 8-10pt, italic for descriptions

## Regenerating Diagrams

To regenerate all diagrams:

```bash
cd book/course-02-programming/ch08-sql-databases/diagrams/
python generate_diagrams.py
```

This will overwrite all existing PNG files with freshly generated versions.

## Updating Content

If diagram locations or names change, update the references in `content.md`:

```markdown
![Diagram Description](diagrams/filename.png)
```

## Source Files

- `generate_diagrams.py` - Main script to generate all diagrams
- `update_content.py` - Script to update content.md with diagram references
- `UPDATE_INSTRUCTIONS.md` - Manual update instructions if needed

## Dependencies

Required Python packages:
- matplotlib
- numpy

These are already part of the standard data science stack used throughout the book.
