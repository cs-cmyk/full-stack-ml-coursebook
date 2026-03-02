> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# 6.1: Python for Data Science: NumPy and Pandas

## Why This Matters

Every data science project starts with data manipulation. Whether analyzing Netflix's billions of viewing records, processing hospital patient data, or building a recommendation engine for e-commerce, tools that can handle millions of rows efficiently are essential. Python's built-in lists are slow—10 to 100 times slower—and lack the operations data scientists need daily. NumPy and Pandas solve this problem. They're the universal foundation: fast, expressive, and industry-standard. If one thing matters from this chapter, it's this: these libraries aren't optional add-ons; they're how data science actually gets done.

## Intuition

### The Spreadsheet Analogy

Think of Pandas as Excel on steroids. In Excel, clicking and dragging filters rows, manually summing columns, and using VLOOKUP merges sheets. With Pandas, code does the same things, but now it's reproducible (save the script, rerun anytime), scalable (works on 10 rows or 10 million rows), and programmable (combine with loops, functions, and APIs).

### The Speed Problem

Imagine a spreadsheet column with 1 million temperatures in Celsius that need conversion to Fahrenheit. In regular Python, a loop would be written:

```python
# The slow way
temperatures_f = []
for temp_c in temperatures_celsius:
    temperatures_f.append(temp_c * 9/5 + 32)
```

This takes about half a second—not terrible for one column, but unacceptable when processing hundreds of columns across millions of rows. With NumPy, the entire array is treated as a single entity:

```python
# The NumPy way - 100x faster
temperatures_f = temperatures_celsius * 9/5 + 32
```

This runs in 0.005 seconds. Why? NumPy operations execute in compiled C code, not interpreted Python. The whole column is processed at once (vectorization), not line-by-line.

### The Restaurant Analogy for Groupby

Imagine a restaurant at the end of the day with a list of orders. The manager asks, "How much did each waiter sell today?"

The natural approach: split the orders by waiter (Alice's orders, Bob's orders), sum the prices for each group, then combine the results into a summary table. This is the **split-apply-combine pattern**, the most powerful concept in data analysis. Pandas makes this trivial:

```python
orders.groupby('waiter')['price'].sum()
```

This single line replaces dozens of manual steps needed in Excel or pure Python.

## Formal Definition

### NumPy Arrays

A **NumPy array** is a homogeneous, multi-dimensional container for numerical data with a fixed size at creation. Unlike Python lists, all elements must have the same data type (dtype), enabling efficient storage and vectorized operations. An array is characterized by its **shape** (dimensions), **dtype** (data type), and **ndim** (number of dimensions).

Notation:
- Let **A** ∈ ℝ^(n × p) be an array with n rows and p columns
- Element access: A[i, j] for row i, column j (zero-indexed)
- Slicing: A[:, j] selects column j; A[i, :] selects row i

### Vectorization

**Vectorization** is the process of applying operations to entire arrays at once, without explicit Python loops. Mathematically, if **v** ∈ ℝ^n and f: ℝ → ℝ, then f(**v**) = [f(v₁), f(v₂), ..., f(vₙ)]. NumPy implements this in optimized C code, providing 10-100× speedups over Python loops.

### Pandas DataFrames

A **DataFrame** is a two-dimensional labeled data structure with columns of potentially different types. It's like a dictionary of Series objects (one-dimensional labeled arrays), where each column is a Series. A DataFrame has an **index** (row labels), **columns** (column labels), and **values** (the actual data).

Notation:
- **df** denotes a DataFrame
- df['column_name'] selects a single column (returns a Series)
- df[['col1', 'col2']] selects multiple columns (returns a DataFrame)
- df.loc[label] uses label-based indexing (inclusive slicing)
- df.iloc[position] uses integer position-based indexing (exclusive slicing)

### The Split-Apply-Combine Pattern

The **split-apply-combine pattern** divides data into groups based on one or more keys, applies a function to each group independently, then combines results into a single data structure:

1. **Split**: Partition data into groups: G₁, G₂, ..., Gₖ
2. **Apply**: Compute function f on each group: f(G₁), f(G₂), ..., f(Gₖ)
3. **Combine**: Aggregate results into output structure

This maps to SQL's GROUP BY and is implemented in Pandas via the `.groupby()` method.

> **Key Concept:** NumPy provides fast numerical arrays with vectorized operations; Pandas builds on NumPy to provide labeled, heterogeneous data structures with powerful manipulation tools. Together, they form the foundation of data science in Python.

## Visualization

Below is a performance comparison showing why vectorization matters. The chart compares Python loops (red) versus NumPy vectorized operations (green) for common operations on 1 million numbers:

```python
import numpy as np
import matplotlib.pyplot as plt
import time

# Create test data
arr = np.random.randn(1_000_000)

# Benchmark function
def benchmark_operation(name, loop_func, vectorized_func):
    # Loop approach
    start = time.time()
    loop_func()
    loop_time = time.time() - start

    # Vectorized approach
    start = time.time()
    vectorized_func()
    vectorized_time = time.time() - start

    return loop_time, vectorized_time, loop_time / vectorized_time

# Operations to test
operations = {}

# Square
operations['Square'] = benchmark_operation(
    'Square',
    lambda: [x**2 for x in arr],
    lambda: arr ** 2
)

# Square root
operations['Sqrt'] = benchmark_operation(
    'Sqrt',
    lambda: [x**0.5 if x > 0 else 0 for x in arr],
    lambda: np.sqrt(np.abs(arr))
)

# Add constant
operations['Add 10'] = benchmark_operation(
    'Add 10',
    lambda: [x + 10 for x in arr],
    lambda: arr + 10
)

# Multiply
operations['Multiply by 2'] = benchmark_operation(
    'Multiply by 2',
    lambda: [x * 2 for x in arr],
    lambda: arr * 2
)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
names = list(operations.keys())
loop_times = [operations[name][0] * 1000 for name in names]  # Convert to ms
vectorized_times = [operations[name][1] * 1000 for name in names]  # Convert to ms
speedups = [operations[name][2] for name in names]

x = np.arange(len(names))
width = 0.35

bars1 = ax.bar(x - width/2, loop_times, width, label='Python Loop', color='#e74c3c')
bars2 = ax.bar(x + width/2, vectorized_times, width, label='NumPy Vectorized', color='#2ecc71')

# Add speedup annotations
for i, (bar, speedup) in enumerate(zip(bars2, speedups)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{speedup:.0f}×\nfaster',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Time (milliseconds, log scale)', fontsize=11)
ax.set_xlabel('Operation', fontsize=11)
ax.set_title('Vectorization Performance: Python Loops vs NumPy\n(1 Million Elements)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('vectorization_performance.png', dpi=100, bbox_inches='tight')
plt.close()

# Output summary
print("Performance Comparison (1 million elements):")
print("-" * 60)
for name in names:
    loop_t, vec_t, speedup = operations[name]
    print(f"{name:15} | Loop: {loop_t*1000:6.1f}ms | NumPy: {vec_t*1000:5.2f}ms | {speedup:4.0f}× faster")
# Output:
# Performance Comparison (1 million elements):
# ------------------------------------------------------------
# Square          | Loop:  421.3ms | NumPy:  4.56ms |   92× faster
# Sqrt            | Loop:  634.2ms | NumPy:  8.23ms |   77× faster
# Add 10          | Loop:  267.8ms | NumPy:  2.14ms |  125× faster
# Multiply by 2   | Loop:  289.5ms | NumPy:  2.08ms |  139× faster
```

**Caption:** This is why NumPy is used: vectorized operations run 50-150× faster than Python loops by executing in compiled C code.

## Examples

### Part 1: NumPy Vectorization Speed Demonstration

```python
# NumPy Vectorization Speed Demonstration
import numpy as np
import time

# Create a large array
n = 1_000_000
arr = np.random.randn(n)

# Method 1: Python loop (slow)
start = time.time()
result_loop = []
for x in arr:
    result_loop.append(x ** 2)
loop_time = time.time() - start

# Method 2: NumPy vectorization (fast)
start = time.time()
result_vectorized = arr ** 2
vectorized_time = time.time() - start

print(f"\nSquaring {n:,} numbers:")
print(f"  Python loop:       {loop_time:.4f} seconds")
print(f"  NumPy vectorized:  {vectorized_time:.4f} seconds")
print(f"  Speedup:           {loop_time/vectorized_time:.0f}× faster")
# Output:
# Squaring 1,000,000 numbers:
#   Python loop:       0.4213 seconds
#   NumPy vectorized:  0.0046 seconds
#   Speedup:           92× faster
```

The code creates an array of 1 million random numbers with `np.random.randn(n)`, generating numbers from a standard normal distribution (mean 0, standard deviation 1).

Two approaches square all numbers:
1. **Python loop**: Iterates through each element, squares it, and appends to a list. Takes ~0.42 seconds.
2. **NumPy vectorized**: The expression `arr ** 2` squares all elements at once. Takes ~0.005 seconds.

The 92× speedup is visceral and memorable. This is the fundamental reason NumPy exists: operations execute in compiled C code, processing entire arrays without Python's interpreter overhead.

### Part 2: NumPy Array Basics

```python
# NumPy Array Fundamentals
import numpy as np

# Creating arrays
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

print(f"\n1D Array: {arr1d}")
print(f"  Shape: {arr1d.shape}, Dimensions: {arr1d.ndim}, Size: {arr1d.size}")
# Output: Shape: (5,), Dimensions: 1, Size: 5

print(f"\n2D Array:\n{arr2d}")
print(f"  Shape: {arr2d.shape}, Dimensions: {arr2d.ndim}, Size: {arr2d.size}")
# Output: Shape: (2, 3), Dimensions: 2, Size: 6

# Array operations (vectorized)
print(f"\nVectorized operations:")
print(f"  arr1d + 10:       {arr1d + 10}")        # [11 12 13 14 15]
print(f"  arr1d * 2:        {arr1d * 2}")         # [ 2  4  6  8 10]
print(f"  arr1d ** 2:       {arr1d ** 2}")        # [ 1  4  9 16 25]
print(f"  np.sqrt(arr1d):   {np.sqrt(arr1d)}")    # [1.  1.41 1.73 2.  2.24]

# Boolean indexing
print(f"\nBoolean indexing:")
mask = arr1d > 3
print(f"  arr1d > 3:        {mask}")              # [False False False  True  True]
print(f"  arr1d[arr1d > 3]: {arr1d[arr1d > 3]}") # [4 5]
```

Two arrays are created:
- `arr1d`: A 1D array with shape (5,), meaning 5 elements in a single dimension
- `arr2d`: A 2D array with shape (2, 3), meaning 2 rows and 3 columns

The `.shape` attribute tells dimensions, `.ndim` tells number of dimensions (1 or 2 here), and `.size` tells total elements.

Vectorized operations apply element-wise:
- `arr1d + 10` adds 10 to every element
- `arr1d ** 2` squares every element
- `np.sqrt(arr1d)` takes the square root of every element

Boolean indexing creates a mask: `arr1d > 3` returns a boolean array `[False, False, False, True, True]`. Using this mask to select elements: `arr1d[arr1d > 3]` returns `[4, 5]`. This is the foundation of data filtering.

### Part 3: Loading and Exploring Data with Pandas

```python
# Pandas DataFrame Basics
import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Map species codes to names
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['species'].map(species_map)

# Basic exploration
print(f"\nDataFrame shape: {df.shape}")  # (150, 5)
print(f"Columns: {list(df.columns)}")

print("\nFirst 5 rows:")
print(df.head())
# Output:
#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa
# 3           4.6          3.1           1.5          0.2  setosa
# 4           5.0          3.6           1.4          0.2  setosa

print("\nDataFrame info:")
print(df.info())
# Output:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 150 entries, 0 to 149
# Data columns (total 5 columns):
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   sepal_length  150 non-null    float64
#  1   sepal_width   150 non-null    float64
#  2   petal_length  150 non-null    float64
#  3   petal_width   150 non-null    float64
#  4   species       150 non-null    object
# dtypes: float64(4), object(1)

print("\nStatistical summary:")
print(df.describe())
# Output:
#        sepal_length  sepal_width  petal_length  petal_width
# count    150.000000   150.000000    150.000000   150.000000
# mean       5.843333     3.057333      3.758000     1.199333
# std        0.828066     0.435866      1.765298     0.762238
# min        4.300000     2.000000      1.000000     0.100000
# 25%        5.100000     2.800000      1.600000     0.300000
# 50%        5.800000     3.000000      4.350000     1.300000
# 75%        6.400000     3.300000      5.100000     1.800000
# max        7.900000     4.400000      6.900000     2.500000
```

The Iris dataset contains 150 flower measurements across 4 features (sepal length, sepal width, petal length, petal width) and 3 species.

The `as_frame=True` parameter returns a Pandas DataFrame instead of NumPy arrays. Column names are renamed for clarity and species codes (0, 1, 2) are mapped to names ('setosa', 'versicolor', 'virginica').

**Essential exploration methods:**
- `.head()`: Shows first 5 rows (quick preview)
- `.shape`: Returns (150, 5) → 150 rows, 5 columns
- `.info()`: Shows column names, data types, non-null counts (detects missing data)
- `.describe()`: Computes statistics (mean, std, quartiles) for numeric columns

These four methods should be the **first thing run** on any new dataset. They answer: What does the data look like? What types are the columns? Are there missing values? What's the statistical distribution?

### Part 4: Data Selection and Filtering

```python
# Selecting and Filtering Data

# Selecting columns
print("\nSingle column (returns Series):")
print(df['sepal_length'].head())
# Output:
# 0    5.1
# 1    4.9
# 2    4.7
# 3    4.6
# 4    5.0

# Multiple columns (returns DataFrame)
print("\nMultiple columns:")
print(df[['sepal_length', 'species']].head())
# Output:
#    sepal_length  species
# 0           5.1   setosa
# 1           4.9   setosa
# 2           4.7   setosa

# Boolean filtering
print("\nFiltering: sepal_length > 7.0")
large_sepals = df[df['sepal_length'] > 7.0]
print(f"Found {len(large_sepals)} flowers")
print(large_sepals[['sepal_length', 'species']].head())
# Output:
# Found 12 flowers
#     sepal_length    species
# 102          7.1  virginica
# 103          7.6  virginica
# ...

# Complex filtering (multiple conditions)
print("\nComplex filter: sepal_length > 7.0 AND species == 'virginica'")
complex_filter = df[(df['sepal_length'] > 7.0) & (df['species'] == 'virginica')]
print(f"Found {len(complex_filter)} flowers")
# Output: Found 12 flowers
```

**Selecting columns:**
- `df['sepal_length']` returns a **Series** (1D labeled array)
- `df[['sepal_length', 'species']]` returns a **DataFrame** (2D labeled structure)

Notice the double brackets in the second case—this tells Pandas that multiple columns are wanted.

**Boolean filtering:**
- `df['sepal_length'] > 7.0` creates a boolean mask (True/False for each row)
- `df[df['sepal_length'] > 7.0]` applies the mask, returning only rows where the condition is True

For complex conditions, use `&` (and) or `|` (or), and **always wrap each condition in parentheses**:
```python
df[(df['sepal_length'] > 7.0) & (df['species'] == 'virginica')]
```

This is SQL's `WHERE` clause, but in Python.

### Part 5: Groupby and Split-Apply-Combine

```python
# Groupby and Split-Apply-Combine

# Simple groupby: mean of all numeric columns by species
print("\nMean measurements by species:")
species_means = df.groupby('species').mean()
print(species_means)
# Output:
#             sepal_length  sepal_width  petal_length  petal_width
# species
# setosa             5.006        3.428         1.462        0.246
# versicolor         5.936        2.770         4.260        1.326
# virginica          6.588        2.974         5.552        2.026

# Multiple aggregations
print("\nMultiple aggregations by species:")
agg_result = df.groupby('species').agg({
    'sepal_length': ['mean', 'min', 'max'],
    'petal_length': ['mean', 'std'],
    'petal_width': 'count'
})
print(agg_result)
# Output:
#            sepal_length              petal_length        petal_width
#                    mean  min  max          mean       std       count
# species
# setosa            5.006  4.3  5.8         1.462  0.173664          50
# versicolor        5.936  4.9  7.0         4.260  0.469911          50
# virginica         6.588  4.9  7.9         5.552  0.551895          50

# Finding patterns: which species has longest petals on average?
print("\nRanking species by average petal length:")
petal_ranking = df.groupby('species')['petal_length'].mean().sort_values(ascending=False)
print(petal_ranking)
# Output:
# species
# virginica     5.552
# versicolor    4.260
# setosa        1.462
```

The groupby operation is the most powerful pattern in data analysis. Here's what happens:

```python
df.groupby('species').mean()
```

1. **Split**: Pandas divides the DataFrame into 3 groups (setosa, versicolor, virginica)
2. **Apply**: It computes the mean of all numeric columns for each group independently
3. **Combine**: It assembles the results into a new DataFrame with species as the index

The output reveals the pattern: virginica has the longest petals (5.55 cm average), setosa has the shortest (1.46 cm), and versicolor is in between (4.26 cm).

For multiple aggregations, `.agg()` is used with a dictionary:
```python
df.groupby('species').agg({
    'sepal_length': ['mean', 'min', 'max'],
    'petal_length': ['mean', 'std']
})
```

This computes different statistics for different columns in a single operation. The result has a hierarchical column structure (MultiIndex) showing which aggregation was applied to which column.

Finally, sorting results finds rankings:
```python
df.groupby('species')['petal_length'].mean().sort_values(ascending=False)
```

This chains operations: group by species, compute mean petal length, then sort descending. Method chaining creates readable data pipelines.

**Summary of examples:**
- NumPy vectorization is 50-100× faster than loops
- Pandas DataFrames organize labeled, heterogeneous data
- Boolean filtering selects subsets of data
- Groupby reveals patterns through aggregation

## Common Pitfalls

### 1. Looping Over DataFrames Instead of Vectorizing

**What beginners do wrong:**
```python
# SLOW - avoid this!
for i in range(len(df)):
    df.loc[i, 'new_col'] = df.loc[i, 'sepal_length'] * 2
```

**Why it's wrong:** This loops through 150 rows in interpreted Python. On a dataset with 1 million rows, this could take minutes instead of milliseconds.

**What to do instead:**
```python
# FAST - vectorized operation
df['new_col'] = df['sepal_length'] * 2
```

This applies the operation to the entire column at once in compiled C code. **Mantra: If writing a for loop over a DataFrame, stop and think—there's almost always a vectorized way.**

### 2. Using `and`/`or` Instead of `&`/`|` for Boolean Filtering

**What beginners do wrong:**
```python
# ERROR - this will fail!
df[df['sepal_length'] > 7.0 and df['species'] == 'virginica']
```

**Why it fails:** The `and` operator works on single boolean values, not arrays. Pandas needs element-wise operators.

**What to do instead:**
```python
# Correct - use & and wrap conditions in parentheses
df[(df['sepal_length'] > 7.0) & (df['species'] == 'virginica')]
```

**Why parentheses matter:** Without them, Python's operator precedence evaluates comparisons incorrectly. Always use parentheses around each condition.

**Alternative (more readable for complex queries):**
```python
# Using query() allows 'and'
df.query('sepal_length > 7.0 and species == "virginica"')
```

### 3. Confusing `.loc` and `.iloc`

**The difference:**
- `.loc[row_label, column_label]`: Uses **labels** (index names, column names)
- `.iloc[row_position, column_position]`: Uses **integer positions**

**The critical slicing difference:**
- `.loc[1:3]` includes rows labeled 1, 2, **and 3** (inclusive)
- `.iloc[1:3]` includes rows at positions 1 and 2, **not 3** (exclusive, like Python slicing)

**Example:**
```python
# Create DataFrame with custom index
df_custom = df.copy()
df_custom.index = ['A', 'B', 'C', 'D', 'E']

# .loc uses labels
df_custom.loc['A':'C']  # Returns rows A, B, C (3 rows)

# .iloc uses positions
df_custom.iloc[0:3]  # Returns rows at positions 0, 1, 2 (3 rows, same result here)

# But if index is numeric, behavior differs:
df.loc[1:3]   # Rows with index labels 1, 2, 3 (3 rows)
df.iloc[1:3]  # Rows at positions 1, 2 (2 rows)
```

**When to use each:**
- Use `.loc` when row/column names are known or for boolean filtering
- Use `.iloc` when positional access is needed (e.g., "first 10 rows")

## Practice

**Practice 1: NumPy Array Basics and Vectorization**

A week's worth of daily high temperatures in Fahrenheit:

```python
import numpy as np
temperatures = np.array([72, 75, 68, 70, 73, 71, 69])
```

1. Convert all temperatures from Fahrenheit to Celsius using the formula: C = (F - 32) × 5/9
2. Find the mean temperature in Celsius (rounded to 1 decimal place)
3. Find the maximum and minimum temperatures in Celsius
4. Create a boolean array indicating which days were above 70°F
5. Use boolean indexing to select only the temperatures above 70°F
6. Calculate how many days were above 70°F

**Expected output:**
```
Temperatures in Celsius: [22.2 23.9 20.0 21.1 22.8 21.7 20.6]
Mean: 21.8°C
Max: 23.9°C, Min: 20.0°C
Days above 70°F: [True True False False True True False]
Temperatures above 70°F: [72 75 73 71]
Count of days above 70°F: 4
```

---

**Practice 2: DataFrame Exploration, Cleaning, and Groupby**

Load the Wine dataset from sklearn and perform a complete exploration and analysis pipeline:

```python
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

# Load data
wine = load_wine(as_frame=True)
df = wine.frame
```

**Part 1: Exploration**
1. Display the shape, column names, and data types
2. Show the first 5 rows
3. Generate a statistical summary with `.describe()`
4. Rename the 'target' column to 'wine_class'
5. Create a new column 'quality_label' that maps wine_class to: 0→'Class A', 1→'Class B', 2→'Class C'

**Part 2: Data Cleaning**
6. Artificially introduce missing values: randomly set 5 values in the 'alcohol' column to NaN using:
   ```python
   indices = np.random.choice(len(df), size=5, replace=False)
   df.loc[indices, 'alcohol'] = np.nan
   ```
7. Detect and count missing values in each column
8. Fill missing values in 'alcohol' with the column mean
9. Verify that no missing values remain

**Part 3: Analysis**
10. Filter to show only wines with alcohol content > 13%
11. Count how many wines are in each wine_class
12. Calculate the mean alcohol content for each wine_class using groupby
13. Find which wine_class has the highest average 'flavanoids' content

---

**Practice 3: Real-World Data Pipeline with California Housing**

A data analyst for a real estate company needs to load the California Housing dataset and perform a comprehensive analysis to identify high-value neighborhoods:

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

# Load data
housing = fetch_california_housing(as_frame=True)
df = housing.frame
```

The dataset contains:
- `MedInc`: Median income in block (in $10,000s)
- `HouseAge`: Median house age in block
- `AveRooms`: Average rooms per household
- `AveBedrms`: Average bedrooms per household
- `Population`: Block population
- `AveOccup`: Average occupancy per household
- `Latitude`, `Longitude`: GPS coordinates
- `MedHouseVal`: Median house value (in $100,000s) - **this is the target**

**Part 1: Data Preparation**
1. Explore the dataset: display shape, columns, check for missing values with `.info()`
2. Create a new categorical column 'price_category' by binning 'MedHouseVal':
   - 'Low': < 1.5
   - 'Medium': 1.5 to 3.0
   - 'High': > 3.0
   Use `pd.cut()` with bins `[0, 1.5, 3.0, 10]` and labels `['Low', 'Medium', 'High']`
3. Create a new column 'rooms_per_household': `AveRooms`
4. Create a new column 'bedrooms_per_room': `AveBedrms / AveRooms` (ratio showing bedroom density)
5. Show the first 5 rows with only the new columns created

**Part 2: Analysis**
6. Group by 'price_category' and calculate:
   - Mean median income
   - Mean house age
   - Mean rooms per household
   - Count of houses in each category
7. What pattern is observed about income and price category?
8. Filter to show only houses in the 'High' price category where `MedInc > 5`
9. How many such houses exist?
10. Find the top 10 most expensive blocks (highest `MedHouseVal`)
11. For these top 10 blocks, what's the average median income?

**Part 3: Advanced Challenge**
12. Create a new DataFrame `income_stats` that shows mean and std of `MedHouseVal` for each `price_category`
13. Which price category has the highest variability (std) in house values?
14. Create a new column in the original DataFrame called 'value_above_category_mean' that shows the difference between each house's value and the mean for its price category
    - Hint: Use `df.groupby('price_category')['MedHouseVal'].transform('mean')` to broadcast category means back to original rows
15. Identify houses that are priced significantly above their category average (>50% above the mean)
16. What percentage of houses fall into this "significantly overpriced" category?

**Hints:**
- For task 14, use: `df.groupby('price_category')['MedHouseVal'].transform('mean')` to broadcast category means back to original rows
- For task 15, use boolean filtering with the column from task 14
- This exercise simulates a real analyst workflow: clean → transform → aggregate → identify outliers

## Solutions

**Solution 1: NumPy Array Basics and Vectorization**

```python
import numpy as np

# Create temperatures array
temperatures = np.array([72, 75, 68, 70, 73, 71, 69])

# 1. Convert to Celsius
temperatures_c = (temperatures - 32) * 5/9
print(f"Temperatures in Celsius: {np.round(temperatures_c, 1)}")

# 2. Mean temperature
mean_temp = np.round(temperatures_c.mean(), 1)
print(f"Mean: {mean_temp}°C")

# 3. Max and min
max_temp = temperatures_c.max()
min_temp = temperatures_c.min()
print(f"Max: {np.round(max_temp, 1)}°C, Min: {np.round(min_temp, 1)}°C")

# 4. Boolean array for temperatures above 70°F
above_70 = temperatures > 70
print(f"Days above 70°F: {above_70}")

# 5. Select temperatures above 70°F
temps_above_70 = temperatures[above_70]
print(f"Temperatures above 70°F: {temps_above_70}")

# 6. Count days above 70°F
count_above_70 = above_70.sum()  # True=1, False=0
print(f"Count of days above 70°F: {count_above_70}")
```

The key insight: vectorized operations like `(temperatures - 32) * 5/9` apply to all elements simultaneously. Boolean arrays can be used both for filtering and counting (since True=1).

---

**Solution 2: DataFrame Exploration, Cleaning, and Groupby**

```python
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

# Load data
wine = load_wine(as_frame=True)
df = wine.frame

# Part 1: Exploration
# 1. Shape, columns, dtypes
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.dtypes)

# 2. First 5 rows
print(df.head())

# 3. Statistical summary
print(df.describe())

# 4. Rename target column
df = df.rename(columns={'target': 'wine_class'})

# 5. Create quality_label column
quality_map = {0: 'Class A', 1: 'Class B', 2: 'Class C'}
df['quality_label'] = df['wine_class'].map(quality_map)

# Part 2: Data Cleaning
# 6. Introduce missing values
np.random.seed(42)  # For reproducibility
indices = np.random.choice(len(df), size=5, replace=False)
df.loc[indices, 'alcohol'] = np.nan

# 7. Detect missing values
print(df.isnull().sum())

# 8. Fill missing values with mean
alcohol_mean = df['alcohol'].mean()
df['alcohol'] = df['alcohol'].fillna(alcohol_mean)

# 9. Verify no missing values
print(f"Missing values after filling: {df.isnull().sum().sum()}")

# Part 3: Analysis
# 10. Filter for alcohol > 13%
high_alcohol = df[df['alcohol'] > 13]
print(f"Wines with alcohol > 13%: {len(high_alcohol)}")

# 11. Count wines in each class
print(df['wine_class'].value_counts())

# 12. Mean alcohol by wine_class
print(df.groupby('wine_class')['alcohol'].mean())

# 13. Highest average flavanoids
flavanoids_by_class = df.groupby('wine_class')['flavanoids'].mean()
highest_class = flavanoids_by_class.idxmax()
print(f"Wine class with highest flavanoids: {highest_class}")
```

**Approach:** Standard data science workflow: explore → clean → analyze. The `.fillna()` method handles missing values. The `.idxmax()` method returns the index (wine_class) with the maximum value.

---

**Solution 3: Real-World Data Pipeline with California Housing**

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

# Load data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Part 1: Data Preparation
# 1. Explore dataset
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.info())

# 2. Create price_category
bins = [0, 1.5, 3.0, 10]
labels = ['Low', 'Medium', 'High']
df['price_category'] = pd.cut(df['MedHouseVal'], bins=bins, labels=labels)

# 3. Create rooms_per_household (already exists as AveRooms)
df['rooms_per_household'] = df['AveRooms']

# 4. Create bedrooms_per_room
df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']

# 5. Show new columns
print(df[['price_category', 'rooms_per_household', 'bedrooms_per_room']].head())

# Part 2: Analysis
# 6. Group by price_category
category_stats = df.groupby('price_category').agg({
    'MedInc': 'mean',
    'HouseAge': 'mean',
    'rooms_per_household': 'mean',
    'MedHouseVal': 'count'
})
category_stats.columns = ['Mean Income', 'Mean Age', 'Mean Rooms', 'Count']
print(category_stats)

# 7. Pattern: Income rises with price category
print("Pattern: Higher income correlates with higher price categories")

# 8. Filter High price category with MedInc > 5
high_income_high_price = df[(df['price_category'] == 'High') & (df['MedInc'] > 5)]
print(f"High price, high income houses: {len(high_income_high_price)}")

# 9. Count (answered above)

# 10. Top 10 most expensive blocks
top_10 = df.nlargest(10, 'MedHouseVal')
print(top_10[['MedHouseVal', 'MedInc', 'Latitude', 'Longitude']])

# 11. Average income of top 10
avg_income_top10 = top_10['MedInc'].mean()
print(f"Average income of top 10 blocks: ${avg_income_top10 * 10000:.0f}")

# Part 3: Advanced Challenge
# 12. Income stats by price category
income_stats = df.groupby('price_category')['MedHouseVal'].agg(['mean', 'std'])
print(income_stats)

# 13. Highest variability
highest_var = income_stats['std'].idxmax()
print(f"Highest variability: {highest_var}")

# 14. Value above category mean
category_means = df.groupby('price_category')['MedHouseVal'].transform('mean')
df['value_above_category_mean'] = df['MedHouseVal'] - category_means

# 15. Significantly overpriced (>50% above mean)
threshold = 0.5
df['overpriced'] = df['value_above_category_mean'] > (category_means * threshold)
overpriced = df[df['overpriced']]

# 16. Percentage overpriced
percentage_overpriced = (len(overpriced) / len(df)) * 100
print(f"Percentage overpriced: {percentage_overpriced:.1f}%")
```

**Approach:**
- `pd.cut()` bins continuous values into categories
- `.transform('mean')` broadcasts group statistics back to original DataFrame length
- Boolean comparison creates the overpriced flag
- The workflow simulates real data science: engineer features → aggregate → merge statistics → identify outliers

---

## Key Takeaways

- **NumPy provides vectorized operations that are 10-100× faster than Python loops** by executing in compiled C code. Always prefer vectorization over explicit loops when working with arrays.

- **Pandas DataFrames are the universal data structure for data science**, combining the familiarity of spreadsheets with the power of programming. Every real-world data science project uses DataFrames.

- **The split-apply-combine pattern (groupby) is the most powerful concept in data analysis**, enabling answering "what's the average X by Y?" questions in a single line of code.

- **Data cleaning isn't optional—it's the first step** in every analysis. Use `.info()`, `.isnull().sum()`, and `.describe()` to detect issues before proceeding.

- **Boolean filtering with `&` and `|` (not `and`/`or`) selects data subsets** based on conditions. Always wrap conditions in parentheses.

- **Understanding the difference between `.loc` (label-based) and `.iloc` (position-based) indexing** is essential, especially for slicing: `.loc` is inclusive, `.iloc` is exclusive.

- **These libraries are not academic exercises—they're industry standard tools**. Netflix uses Pandas to analyze billions of viewing records; financial firms use NumPy for risk modeling. Learning them well opens the door to every data science job.

**Next:** [Chapter X.Y] covers data visualization with Matplotlib and Seaborn to communicate insights visually.
