"""
Script to update content.md with diagram references
Replaces ASCII art with image references to generated diagrams
"""

def update_content_md():
    content_path = '/home/chirag/ds-book/book/course-02-programming/ch08-sql-databases/content.md'

    # Read the current content
    with open(content_path, 'r') as f:
        content = f.read()

    # Replace 1: CSV vs Database ASCII art
    old_csv_section = '''## Visual

```
┌─────────────────────────────────────────────────────────────────┐
│                    CSV Files vs Database                        │
├────────────────────────────────┬────────────────────────────────┤
│         CSV Approach           │       Database Approach        │
├────────────────────────────────┼────────────────────────────────┤
│  customers.csv                 │   ┌─────────────────────────┐  │
│  orders.csv                    │   │      DATABASE.db        │  │
│  products.csv                  │   │                         │  │
│                                │   │  ┌─────────────────┐    │  │
│  Python Code:                  │   │  │  customers      │    │  │
│  1. Load all CSVs              │   │  │  (id, name,     │    │  │
│  2. df.merge() manually        │   │  │   email, city)  │    │  │
│  3. Filter in memory           │   │  └────────┬────────┘    │  │
│                                │   │           │             │  │
│  Memory: Load everything       │   │  ┌────────▼────────┐    │  │
│  Speed: Slow for large data    │   │  │    orders       │    │  │
│  Multi-user: File conflicts    │   │  │  (id, cust_id,  │    │  │
│                                │   │  │   date, total)  │    │  │
│                                │   │  └─────────────────┘    │  │
│                                │   │                         │  │
│                                │   │  SQL Query:             │  │
│                                │   │  SELECT ... WHERE ...   │  │
│                                │   │  JOIN ... GROUP BY ...  │  │
│                                │   │                         │  │
│                                │   │  Memory: Load only      │  │
│                                │   │          filtered data  │  │
│                                │   │  Speed: Fast (indexed)  │  │
│                                │   │  Multi-user: Safe       │  │
│                                │   └─────────────────────────┘  │
└────────────────────────────────┴────────────────────────────────┘

When to use CSV: Small datasets, quick sharing, one-time analysis
When to use Database: Large data, relationships, production systems, multi-user
```'''

    new_csv_section = '''## Visual

![CSV Files vs Database](diagrams/csv_vs_database.png)

**When to use CSV:** Small datasets, quick sharing, one-time analysis
**When to use Database:** Large data, relationships, production systems, multi-user'''

    content = content.replace(old_csv_section, new_csv_section)

    # Replace 2: SQL Query Execution Order
    old_execution_section = '''**SQL Query Execution Order:**
```
Written Order:              Actual Execution Order:
--------------              ----------------------
SELECT columns              1. FROM table(s)       ← Get data source
FROM table                  2. WHERE condition     ← Filter rows
WHERE condition             3. GROUP BY columns    ← Group data
GROUP BY columns            4. HAVING condition    ← Filter groups
HAVING condition            5. SELECT columns      ← Choose what to show
ORDER BY columns            6. ORDER BY columns    ← Sort results
LIMIT n                     7. LIMIT n             ← Take first n

⚠️ Understanding this order prevents 90% of SQL errors!
   Example: You can't reference a SELECT alias in WHERE
   because WHERE executes before SELECT.
```'''

    new_execution_section = '''**SQL Query Execution Order:**

![SQL Query Execution Order](diagrams/sql_execution_order.png)'''

    content = content.replace(old_execution_section, new_execution_section)

    # Replace 3: Add WHERE vs HAVING diagram
    old_mental_model = '''**Mental Model:** WHERE is a bouncer at the door (filters rows entering the group). HAVING is a bouncer after groups form (filters the groups themselves).'''

    new_mental_model = '''**Mental Model:** WHERE is a bouncer at the door (filters rows entering the group). HAVING is a bouncer after groups form (filters the groups themselves).

![WHERE vs HAVING Comparison](diagrams/where_vs_having.png)'''

    content = content.replace(old_mental_model, new_mental_model)

    # Replace 4: JOIN Types Visual Guide
    old_join_section = '''### A. JOIN Types Visual Guide

```
Table: guests (RSVP list)          Table: meals (meal selections)
guest_id | name                    guest_id | meal
---------|--------                 ---------|-------
1        | Alice                   1        | Chicken
2        | Bob                     3        | Fish
3        | Carol                   4        | Vegan
4        | David

INNER JOIN (only matches):          LEFT JOIN (all guests + meals):
guest_id | name  | meal             guest_id | name  | meal
---------|-------|-------           ---------|-------|-------
1        | Alice | Chicken          1        | Alice | Chicken
3        | Carol | Fish             2        | Bob   | NULL
4        | David | Vegan            3        | Carol | Fish
                                    4        | David | Vegan

RIGHT JOIN (all meals + guests):    FULL OUTER JOIN (everything):
guest_id | name  | meal             guest_id | name  | meal
---------|-------|-------           ---------|-------|-------
1        | Alice | Chicken          1        | Alice | Chicken
3        | Carol | Fish             2        | Bob   | NULL
4        | David | Vegan            3        | Carol | Fish
                                    4        | David | Vegan

Use INNER when: You only want records that exist in both tables
Use LEFT when: You want all records from the primary table (most common)
Use RIGHT when: Rare—usually just flip the tables and use LEFT instead
Use FULL OUTER when: You want everything from both tables (also rare)
```'''

    new_join_section = '''### A. JOIN Types Visual Guide

![JOIN Types Visual Guide](diagrams/join_types.png)

**Example Tables:**

```
Table: guests (RSVP list)          Table: meals (meal selections)
guest_id | name                    guest_id | meal
---------|--------                 ---------|-------
1        | Alice                   1        | Chicken
2        | Bob                     3        | Fish
3        | Carol                   4        | Vegan
4        | David
```

**When to use each JOIN type:**
- **INNER JOIN**: Only want records that exist in both tables
- **LEFT JOIN**: Want all records from the primary table (most common)
- **RIGHT JOIN**: Rare—usually just flip the tables and use LEFT instead
- **FULL OUTER JOIN**: Want everything from both tables (also rare)'''

    content = content.replace(old_join_section, new_join_section)

    # Replace 5: Add reference to GROUP BY example diagram
    old_groupby_text = '''# Output:
#   Region  NumAreas  AvgIncome  AvgHouseValue  MinPrice  MaxPrice
# 0  North      4145      3.845          2.089     0.150     5.000
# ...'''

    new_groupby_text = '''# Output:
#   Region  NumAreas  AvgIncome  AvgHouseValue  MinPrice  MaxPrice
# 0  North      4145      3.845          2.089     0.150     5.000
# ...

# Example visualization of GROUP BY results:
# See diagrams/groupby_example.png for a bar chart representation'''

    content = content.replace(old_groupby_text, new_groupby_text)

    # Write the updated content
    with open(content_path, 'w') as f:
        f.write(content)

    print("✅ Successfully updated content.md with diagram references!")
    print()
    print("Changes made:")
    print("  1. ✓ Replaced CSV vs Database ASCII art with image")
    print("  2. ✓ Replaced SQL Query Execution Order ASCII art with image")
    print("  3. ✓ Added WHERE vs HAVING diagram")
    print("  4. ✓ Replaced JOIN Types ASCII art with image")
    print("  5. ✓ Added GROUP BY example diagram reference")
    print()
    print("All diagrams are now properly referenced in content.md")

if __name__ == '__main__':
    update_content_md()
