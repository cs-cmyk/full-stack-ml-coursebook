# Content.md Update Instructions

This document describes the changes needed to replace ASCII art with generated diagrams.

## Changes to Make

### 1. Replace CSV vs Database ASCII Art (Lines 68-102)

**Original:**
```
```
┌─────────────────────────────────────────────────────────────────┐
│                    CSV Files vs Database                        │
[... ASCII art ...]
└────────────────────────────────────────────────────────────────┘

When to use CSV: Small datasets, quick sharing, one-time analysis
When to use Database: Large data, relationships, production systems, multi-user
```
```

**Replace with:**
```markdown
![CSV Files vs Database](diagrams/csv_vs_database.png)

**When to use CSV:** Small datasets, quick sharing, one-time analysis
**When to use Database:** Large data, relationships, production systems, multi-user
```

---

### 2. Replace SQL Query Execution Order ASCII Art (Lines 104-119)

**Original:**
```
**SQL Query Execution Order:**
```
Written Order:              Actual Execution Order:
--------------              ----------------------
[... ASCII art ...]
```
```

**Replace with:**
```markdown
**SQL Query Execution Order:**

![SQL Query Execution Order](diagrams/sql_execution_order.png)
```

---

### 3. Add GROUP BY Example Diagram Reference (After line 285)

**Current line 285:**
```python
plt.savefig('sql_groupby_results.png', dpi=100, bbox_inches='tight')
```

**Add after the code block (around line 313):**
```markdown
**Example Output:**

![GROUP BY Results](diagrams/groupby_example.png)
```

---

### 4. Add WHERE vs HAVING Diagram (After line 374, in Common Pitfalls section)

**After the "Mental Model" line (around line 374), add:**
```markdown

![WHERE vs HAVING Comparison](diagrams/where_vs_having.png)
```

---

### 5. Replace JOIN Types ASCII Art (Lines 696-727)

**Original:**
```
### A. JOIN Types Visual Guide

```
Table: guests (RSVP list)          Table: meals (meal selections)
[... ASCII art showing different JOIN types ...]
```
```

**Replace with:**
```markdown
### A. JOIN Types Visual Guide

![JOIN Types Visual Guide](diagrams/join_types.png)

**Table Examples:**

Table: guests (RSVP list)          Table: meals (meal selections)
guest_id | name                    guest_id | meal
---------|--------                 ---------|-------
1        | Alice                   1        | Chicken
2        | Bob                     3        | Fish
3        | Carol                   4        | Vegan
4        | David

**Use Cases:**
- **INNER JOIN**: When you only want records that exist in both tables
- **LEFT JOIN**: When you want all records from the primary table (most common)
- **RIGHT JOIN**: Rare—usually just flip the tables and use LEFT instead
- **FULL OUTER JOIN**: When you want everything from both tables (also rare)
```

---

## Generated Diagrams

All diagrams have been generated and saved to:
`book/course-02-programming/ch08-sql-databases/diagrams/`

### Files Created:
1. `csv_vs_database.png` - Side-by-side comparison of CSV vs Database approaches
2. `sql_execution_order.png` - Visual flowchart showing SQL query execution order
3. `groupby_example.png` - Bar chart showing average house values by region
4. `where_vs_having.png` - Visual comparison of WHERE (row filter) vs HAVING (group filter)
5. `join_types.png` - Four-panel diagram showing INNER, LEFT, RIGHT, and FULL OUTER joins

### Diagram Specifications:
- Resolution: 150 DPI
- Max width: ~800px (varies by diagram)
- Background: White
- Color palette:
  - Blue (#2196F3)
  - Green (#4CAF50)
  - Orange (#FF9800)
  - Red (#F44336)
  - Purple (#9C27B0)
  - Gray (#607D8B)
- Font size: Minimum 12pt for readability
- All diagrams use tight_layout() for clean borders
