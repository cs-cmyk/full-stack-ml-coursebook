> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# Chapter 14: Categorical Features

## Why This Matters

Most real-world datasets contain categorical features—customer segments, product types, cities, medical diagnoses. Yet machine learning models require numeric inputs. Simply assigning numbers like Male=1, Female=2 creates false mathematical relationships that harm model performance. Understanding how to properly encode categorical features is essential for building accurate models with messy, real-world data.

## Intuition

Imagine analyzing ice cream sales where the dataset has a "Flavor" column with values like Vanilla, Chocolate, Strawberry, and Mint. One might think, "Models need numbers, so assign Vanilla=1, Chocolate=2, Strawberry=3, Mint=4." But this creates nonsense relationships:

- The model now thinks Chocolate is "twice as much" as Vanilla
- Strawberry is somehow "between" Chocolate and Mint
- If averaging (Vanilla + Strawberry)/2 = (1+3)/2 = 2 = Chocolate?

This is absurd! Flavors don't have a meaningful numerical order. They're just different categories.

The solution is **one-hot encoding**: instead of one column with arbitrary numbers, create separate yes/no columns for each flavor. For Vanilla, the encoding is [1, 0, 0, 0]. For Chocolate, it's [0, 1, 0, 0]. Now each flavor is independent—no false math relationships. The model can learn that Mint sells better on hot days without incorrectly thinking Mint is "4 times" anything.

But not all categories are created equal. Consider T-shirt sizes: Small, Medium, Large, Extra Large. Here, there *is* a meaningful order: Small < Medium < Large < XL. Assigning S=1, M=2, L=3, XL=4 preserves this relationship. This is called **ordinal encoding**.

Here's a simple test: Can two categories be meaningfully averaged? If averaging Small and Extra Large gives something close to Large, it's **ordinal** (has inherent order). If averaging Red and Green doesn't give Yellow (or anything meaningful in the dataset), it's **nominal** (no order)—use one-hot encoding.

The encoding strategy depends on three factors:
1. **Variable type**: Nominal (no order), ordinal (meaningful order), or binary (two values)
2. **Cardinality**: How many unique categories exist (5 vs. 500)
3. **Model type**: Linear models need different encoding than tree-based models

Think of it like organizing a filing cabinet. One-hot encoding is like giving each category its own drawer—easy to find, but many drawers are needed. Label encoding is like numbering files—compact, but the numbering might imply false relationships. For rare categories, grouping them into a "Miscellaneous" drawer saves space while preserving information about common items.

## Formal Definition

Let **X** be a feature matrix of dimension n × p, where n is the number of samples and p is the number of features. A **categorical feature** is a feature that takes on one of a finite set of discrete values (categories or levels) without inherent numerical meaning.

**Types of Categorical Variables:**

1. **Nominal**: Categories with no inherent ordering
   - Example: C = {red, blue, green}
   - No meaningful relationship: red ≮ blue

2. **Ordinal**: Categories with meaningful ordering but not necessarily equal spacing
   - Example: E = {high school, bachelor, master, PhD}
   - Ordering exists: high school < bachelor < master < PhD

3. **Binary**: Special case with exactly two categories
   - Example: S = {yes, no} or S = {0, 1}

**Encoding Functions:**

Given a categorical feature **c** ∈ C with K unique categories {c₁, c₂, ..., c_K}, the following are defined:

**One-Hot Encoding** (Dummy Variables):
- φ: C → {0,1}^K or {0,1}^(K-1)
- Maps each category to a binary vector
- For category c_i: φ(c_i) = [0, ..., 0, 1, 0, ..., 0] where the i-th position is 1
- With drop_first=True: Use K-1 columns to avoid multicollinearity

**Label Encoding** (Integer Encoding):
- ψ: C → {0, 1, ..., K-1}
- Maps each category to a unique integer
- No inherent meaning to the numerical ordering (unless ordinal)

**Ordinal Encoding**:
- ω: C → {1, 2, ..., K}
- Maps categories to integers preserving meaningful order
- For ordered categories: c₁ < c₂ < ... < c_K implies ω(c₁) < ω(c₂) < ... < ω(c_K)

> **Key Concept:** Categorical features must be encoded as numbers for machine learning models, but the encoding strategy must preserve the true relationships between categories—avoid creating false ordinal relationships for nominal variables.

## Visualization

```
BEFORE: Original Categorical Data          AFTER: One-Hot Encoded Data
┌─────────┬────────┐                       ┌─────────┬─────────┬──────────┬────────────┐
│ Sample  │ Color  │                       │ Sample  │ is_Red  │ is_Blue  │ is_Green   │
├─────────┼────────┤                       ├─────────┼─────────┼──────────┼────────────┤
│    1    │  Red   │  ──────────────────>  │    1    │    1    │    0     │     0      │
│    2    │  Blue  │                       │    2    │    0    │    1     │     0      │
│    3    │  Green │                       │    3    │    0    │    0     │     1      │
│    4    │  Red   │                       │    4    │    1    │    0     │     0      │
│    5    │  Blue  │                       │    5    │    0    │    1     │     0      │
└─────────┴────────┘                       └─────────┴─────────┴──────────┴────────────┘
    1 column                                        3 binary columns
                                                  (or 2 with drop_first=True)

Key: Each unique category becomes its own binary indicator column.
     Only one column is "1" per row (mutually exclusive categories).
```

**Decision Tree for Encoding Selection:**

```
                    ┌─────────────────────────┐
                    │   Categorical Feature   │
                    └───────────┬─────────────┘
                                │
                  ┌─────────────▼──────────────┐
                  │ Does it have inherent order?│
                  └──────┬─────────────┬────────┘
                         │             │
                    NO   │             │  YES
                         │             │
        ┌────────────────▼──┐      ┌───▼─────────────┐
        │ NOMINAL VARIABLE   │      │ ORDINAL VARIABLE│
        └────────┬───────────┘      └────┬────────────┘
                 │                        │
        ┌────────▼────────┐              │
        │ Is cardinality  │              │
        │     < 15?       │              │
        └──┬──────────┬───┘              │
           │          │                   │
       YES │          │ NO                │
           │          │                   │
    ┌──────▼──┐  ┌────▼─────────┐   ┌────▼─────────────┐
    │ One-Hot │  │ Frequency or │   │ Ordinal Encoding │
    │Encoding │  │ Target or    │   │ (preserve order) │
    │         │  │ Grouping     │   │                  │
    └─────────┘  └──────────────┘   └──────────────────┘

    Special Case:
    Binary (2 categories) → Simple 0/1 encoding (single column)

    Tree-based models: Label encoding acceptable for nominal high-cardinality
```

## Examples

### Part 1: One-Hot Encoding Basics

```python
# Complete Example: Categorical Feature Encoding
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set random state for reproducibility
np.random.seed(42)

# ============================================================
# PART 1: One-Hot Encoding with Wine Dataset
# ============================================================
print("=" * 60)
print("PART 1: One-Hot Encoding Demonstration")
print("=" * 60)

from sklearn.datasets import load_wine

# Load wine dataset
wine = load_wine()
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)

# Convert numeric target to categorical for demonstration
target_names = {0: 'Class_0', 1: 'Class_1', 2: 'Class_2'}
df_wine['wine_type'] = pd.Series(wine.target).map(target_names)

# Take small sample for clarity
df_sample = df_wine[['alcohol', 'malic_acid', 'wine_type']].head(8)

print("\nOriginal data with categorical feature:")
print(df_sample)
print(f"\nShape: {df_sample.shape}")
print(f"wine_type categories: {df_sample['wine_type'].unique()}")

# Method 1: One-hot encoding with pandas
print("\n" + "-" * 60)
print("Method 1: pandas get_dummies()")
print("-" * 60)

df_encoded = pd.get_dummies(df_sample, columns=['wine_type'], drop_first=False)
print("\nOne-hot encoded (keeping all categories):")
print(df_encoded)
print(f"Shape after encoding: {df_encoded.shape}")
print("Note: 3 categories → 3 new columns")

# With drop_first to avoid dummy variable trap
df_encoded_drop = pd.get_dummies(df_sample, columns=['wine_type'], drop_first=True)
print("\nOne-hot encoded (with drop_first=True):")
print(df_encoded_drop)
print(f"Shape after encoding: {df_encoded_drop.shape}")
print("Note: 3 categories → 2 new columns (avoids multicollinearity)")
```

**Part 1 Walkthrough:**

The code starts by loading the Wine dataset and converting its numeric target into categorical labels ('Class_0', 'Class_1', 'Class_2'). This demonstrates the fundamental transformation: a single categorical column expands into multiple binary columns.

With `pd.get_dummies()`, one binary column per category is created. The `drop_first=True` parameter is crucial—it drops one category to avoid the **dummy variable trap**. Why? With an intercept term in linear models, having all dummy variables creates perfect multicollinearity. If it's not Red and not Blue, it must be Green! Using K-1 columns preserves all information while avoiding numerical instability.

### Part 2: sklearn OneHotEncoder

```python
# Method 2: One-hot encoding with sklearn
print("\n" + "-" * 60)
print("Method 2: sklearn OneHotEncoder()")
print("-" * 60)

encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(df_sample[['wine_type']])

print("\nEncoded array:")
print(X_encoded)
print(f"Shape: {X_encoded.shape}")
print(f"Categories learned: {encoder.categories_}")
print(f"Feature names: {encoder.get_feature_names_out()}")
```

**Part 2 Walkthrough:**

The sklearn `OneHotEncoder` provides more control for production pipelines. The `sparse_output=False` returns a dense array (easier to visualize), while `sparse_output=True` (default) saves memory for high-cardinality features. The `handle_unknown='ignore'` parameter is critical for deployment—when a test sample has a category never seen during training (like a new city), it creates an all-zero row rather than erroring.

### Part 3: Encoding Strategy by Variable Type

```python
# ============================================================
# PART 2: Comparing Encoding Types
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Encoding Types Comparison")
print("=" * 60)

# Create synthetic customer dataset
df_customers = pd.DataFrame({
    'customer_id': range(1, 11),
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'NYC', 'LA', 'Chicago', 'Houston', 'NYC', 'LA'],
    'education': ['HS', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'HS', 'Master', 'Bachelor', 'PhD', 'Master'],
    'subscribed': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'purchase_amount': [100, 50, 200, 300, 75, 150, 80, 250, 350, 120]
})

print("\nOriginal customer data:")
print(df_customers)

# Nominal: One-hot encode 'city'
print("\n" + "-" * 60)
print("Encoding NOMINAL feature (city):")
print("-" * 60)
print("Strategy: One-hot encoding (no inherent order)")

df_city_encoded = pd.get_dummies(df_customers['city'], prefix='city', drop_first=True)
print("\nCity one-hot encoded:")
print(df_city_encoded.head())
```

**Part 3 Walkthrough:**

Here a synthetic customer dataset with three different categorical types demonstrates appropriate encoding for each:

1. **City (Nominal)**: No inherent order between NYC, LA, Chicago, Houston. One-hot encoding is used. Assigning NYC=1, LA=2 would falsely imply LA is "twice" NYC or "between" NYC and Chicago—nonsense.

### Part 4: Ordinal and Binary Encoding

```python
# Ordinal: Ordinal encode 'education'
print("\n" + "-" * 60)
print("Encoding ORDINAL feature (education):")
print("-" * 60)
print("Strategy: Ordinal encoding (preserve order: HS < Bachelor < Master < PhD)")

ordinal_encoder = OrdinalEncoder(categories=[['HS', 'Bachelor', 'Master', 'PhD']])
education_encoded = ordinal_encoder.fit_transform(df_customers[['education']])

print("\nEducation ordinal encoded:")
print(pd.DataFrame({
    'original': df_customers['education'],
    'encoded': education_encoded.flatten()
}))
print("Mapping: HS=0, Bachelor=1, Master=2, PhD=3")

# Binary: Simple 0/1 encoding
print("\n" + "-" * 60)
print("Encoding BINARY feature (subscribed):")
print("-" * 60)
print("Strategy: Simple 0/1 encoding (only 2 categories)")

df_customers['subscribed_encoded'] = (df_customers['subscribed'] == 'Yes').astype(int)

print("\nSubscribed binary encoded:")
print(df_customers[['subscribed', 'subscribed_encoded']].head())
```

**Part 4 Walkthrough:**

2. **Education (Ordinal)**: Clear progression: High School < Bachelor < Master < PhD. `OrdinalEncoder` is used with explicit category ordering. The model can now learn that the distance from HS (0) to Bachelor (1) represents a meaningful educational step.

3. **Subscribed (Binary)**: Only two values (Yes/No). Simple 0/1 encoding with a single column is used. No need for two one-hot columns when one binary indicator suffices.

This section reinforces the decision rule: identify the variable type first, then choose the encoding strategy.

### Part 5: High-Cardinality Strategies

```python
# ============================================================
# PART 3: High-Cardinality Encoding Strategies
# ============================================================
print("\n" + "=" * 60)
print("PART 3: High-Cardinality Encoding Strategies")
print("=" * 60)

from sklearn.datasets import fetch_california_housing

# Load California housing data
housing = fetch_california_housing()
df_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
df_housing['target'] = housing.target

# Create synthetic high-cardinality feature (neighborhood_id)
# Based on lat/lon, create ~50 unique neighborhoods
df_housing['neighborhood_id'] = (
    df_housing['Latitude'].round(1).astype(str) + '_' +
    df_housing['Longitude'].round(1).astype(str)
)

print(f"\nDataset shape: {df_housing.shape}")
print(f"Number of unique neighborhoods: {df_housing['neighborhood_id'].nunique()}")
print("\nSample data:")
print(df_housing[['Latitude', 'Longitude', 'neighborhood_id', 'target']].head())

# Strategy 1: Frequency Encoding
print("\n" + "-" * 60)
print("Strategy 1: Frequency Encoding")
print("-" * 60)

frequency_map = df_housing['neighborhood_id'].value_counts(normalize=True).to_dict()
df_housing['neighborhood_frequency'] = df_housing['neighborhood_id'].map(frequency_map)

print("\nTop 5 neighborhoods by frequency:")
print(df_housing.groupby('neighborhood_id').agg({
    'neighborhood_frequency': 'first'
}).sort_values('neighborhood_frequency', ascending=False).head())

print(f"\nResult: 1 feature column (down from {df_housing['neighborhood_id'].nunique()} potential one-hot columns)")
```

**Part 5 Walkthrough:**

California Housing has continuous lat/lon coordinates. A high-cardinality feature is synthesized by binning them into ~52 unique neighborhood IDs. One-hot encoding would create 52 columns—potentially problematic for dimensionality and overfitting.

Three alternatives are demonstrated:

1. **Frequency Encoding**: Replace each neighborhood with its frequency (proportion of dataset). Common neighborhoods get higher values, rare ones get lower values. This captures "how typical is this neighborhood?" in a single column. Trade-off: neighborhoods with identical frequency become indistinguishable.

### Part 6: Grouping Rare Categories

```python
# Strategy 2: Grouping Rare Categories
print("\n" + "-" * 60)
print("Strategy 2: Grouping Rare Categories")
print("-" * 60)

# Keep top 10 neighborhoods, group rest as 'Other'
top_n = 10
top_neighborhoods = df_housing['neighborhood_id'].value_counts().head(top_n).index
df_housing['neighborhood_grouped'] = df_housing['neighborhood_id'].apply(
    lambda x: x if x in top_neighborhoods else 'Other'
)

print(f"\nOriginal categories: {df_housing['neighborhood_id'].nunique()}")
print(f"After grouping: {df_housing['neighborhood_grouped'].nunique()}")
print("\nGrouped category counts:")
print(df_housing['neighborhood_grouped'].value_counts())
```

**Part 6 Walkthrough:**

2. **Grouping Rare Categories**: Keep the top 10 most frequent neighborhoods by name, lump the rest into "Other". Now there are 11 categories instead of 52—much more manageable for one-hot encoding. This works when rare categories don't have unique patterns worth capturing.

### Part 7: Target Encoding

```python
# Strategy 3: Target Encoding (with proper CV to avoid leakage)
print("\n" + "-" * 60)
print("Strategy 3: Target Encoding (Leave-One-Out)")
print("-" * 60)

# Compute target mean for each neighborhood (leave-one-out)
def target_encode_loo(df, cat_col, target_col):
    """Leave-one-out target encoding to prevent leakage"""
    global_mean = df[target_col].mean()

    # For each category, compute mean excluding current sample
    encoded = []
    for idx in df.index:
        cat = df.loc[idx, cat_col]
        # Mean of target for this category, excluding current row
        mask = (df[cat_col] == cat) & (df.index != idx)
        cat_mean = df.loc[mask, target_col].mean() if mask.sum() > 0 else global_mean
        encoded.append(cat_mean)

    return encoded

# Take subset for demonstration (full dataset would be slow)
df_subset = df_housing.sample(n=1000, random_state=42)
df_subset['neighborhood_target_encoded'] = target_encode_loo(
    df_subset, 'neighborhood_id', 'target'
)

print("\nTarget encoding example:")
print(df_subset[['neighborhood_id', 'target', 'neighborhood_target_encoded']].head(10))
print("\nNote: Each neighborhood encoded with mean target value (house price)")
print("Leave-one-out prevents using sample's own target in encoding")
```

**Part 7 Walkthrough:**

3. **Target Encoding (Leave-One-Out)**: Replace each neighborhood with the mean target value (house price) for that neighborhood. Critical implementation detail: **leave-one-out encoding** is used—for each sample, the neighborhood mean is computed excluding that sample's own target value. This prevents direct leakage of the target into the features (which would cause severe overfitting). Without leave-one-out, the model would "cheat" by using information about the target it's trying to predict.

### Part 8: Production Pipeline with ColumnTransformer

```python
# ============================================================
# PART 4: Complete Pipeline with ColumnTransformer
# ============================================================
print("\n" + "=" * 60)
print("PART 4: Production Pipeline with Mixed Data Types")
print("=" * 60)

# Create synthetic classification dataset
np.random.seed(42)
df_pipeline = pd.DataFrame({
    'age': np.random.randint(18, 70, 200),
    'income': np.random.randint(20000, 150000, 200),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 200),
    'education': np.random.choice(['HS', 'Bachelor', 'Master', 'PhD'], 200),
    'subscribed': np.random.choice([0, 1], 200)
})

# Add some missing values
df_pipeline.loc[df_pipeline.sample(10, random_state=42).index, 'age'] = np.nan
df_pipeline.loc[df_pipeline.sample(15, random_state=42).index, 'city'] = np.nan

print("\nDataset for pipeline:")
print(df_pipeline.head(10))
print(f"\nShape: {df_pipeline.shape}")
print(f"Missing values:\n{df_pipeline.isnull().sum()}")

# Define feature types
numeric_features = ['age', 'income']
categorical_features = ['city', 'education']
target = 'subscribed'

# Split data
X = df_pipeline.drop(target, axis=1)
y = df_pipeline[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

**Part 8 Walkthrough:**

This section demonstrates industry best practices for handling mixed data types.

A dataset with both numeric features (age, income) and categorical features (city, education) is created, including missing values to simulate real-world messiness.

### Part 9: Building the Pipeline

```python
# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Fit pipeline (encoding is learned from training data only!)
print("\n" + "-" * 60)
print("Training pipeline...")
print("-" * 60)

pipeline.fit(X_train, y_train)

# Evaluate
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"\nTraining accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")

# Show feature names after encoding
feature_names = (numeric_features +
                 list(pipeline.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .named_steps['onehot']
                      .get_feature_names_out(categorical_features)))

print(f"\nFeature names after encoding:")
for i, name in enumerate(feature_names, 1):
    print(f"  {i}. {name}")
```

**Part 9 Walkthrough:**

The `ColumnTransformer` applies different preprocessing to different column types:
- Numeric pipeline: Impute missing values with median
- Categorical pipeline: Impute missing with most frequent value, then one-hot encode

The full `Pipeline` ensures these transformations are **learned from training data only**. When calling `pipeline.fit(X_train, y_train)`, the SimpleImputer learns the median age from training data, and the OneHotEncoder learns which categories exist from training data. When applying `pipeline.predict(X_test)`, these learned parameters are used—the test set never influences the preprocessing parameters. This prevents data leakage.

The output shows 8 features after encoding: 2 numeric (age, income) + 3 city dummies + 3 education dummies. The model trained successfully and can make predictions on new data.

### Part 10: Handling Unseen Categories

```python
# Demonstrate handling unseen category
print("\n" + "-" * 60)
print("Handling unseen category in test data:")
print("-" * 60)

# Create sample with unseen city
X_new = pd.DataFrame({
    'age': [35],
    'income': [75000],
    'city': ['Miami'],  # Never seen during training!
    'education': ['Bachelor']
})

print("\nNew sample with unseen city 'Miami':")
print(X_new)

try:
    prediction = pipeline.predict(X_new)
    print(f"\nPrediction: {prediction[0]}")
    print("Note: handle_unknown='ignore' creates all-zero encoding for 'Miami'")
except Exception as e:
    print(f"\nError: {e}")

# Output summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key Takeaways:
1. One-hot encoding: 3 categories → 2 columns (with drop_first=True)
2. Ordinal encoding: Preserves meaningful order (HS=0, Bachelor=1, etc.)
3. High cardinality: Use frequency/target/grouping (not one-hot)
4. Pipeline pattern: Fit encoders on train data only (prevents leakage)
5. Handle unseen categories: Use handle_unknown='ignore'
""")
```

**Part 10 Walkthrough:**

Finally, handling an unseen category ("Miami") that wasn't in the training set is demonstrated. Because `handle_unknown='ignore'` was specified, the encoder creates an all-zero encoding for Miami (treating it as "none of the known cities"). Without this parameter, the encoder would raise an error, breaking the production system.

## Common Pitfalls

**1. Label Encoding Nominal Features with Linear Models**

Beginners often assign arbitrary integers to nominal categories (Red=1, Blue=2, Green=3) because "the model needs numbers." But linear models compute weighted sums: if the model learns coefficient w=5 for a color feature, it predicts w × color_value. This means Blue (2) contributes twice as much as Red (1), and Green (3) contributes three times as much. The model is learning nonsense math relationships.

**Why it happens:** It's the simplest encoding to implement—just one line of code. And it works fine with tree-based models (which split on thresholds, not arithmetic), so tutorials using Random Forests don't show the problem.

**What to do instead:** Ask, "Does this feature have inherent order?" If no (nominal), use one-hot encoding for linear models. Label encoding is only appropriate for ordinal features or when using tree-based models that treat splits as category membership tests rather than arithmetic operations.

**2. Encoding Before Train/Test Split (Data Leakage)**

A common mistake is encoding the entire dataset first, then splitting:

```python
# WRONG! Don't do this:
df_encoded = pd.get_dummies(df, columns=['category'])  # Encodes ALL data
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y)
```

This seems harmless for one-hot encoding (categories don't change), but it's catastrophic for target encoding, frequency encoding, or any technique where encoding parameters are learned from data. With target encoding, the mean target value for each category would be computed using the **full dataset including test set**. The model then sees test set information during training—direct leakage.

**What to do instead:** Always split first, fit encoders on training data only:

```python
# CORRECT:
X_train, X_test, y_train, y_test = train_test_split(X, y)
encoder = TargetEncoder()
encoder.fit(X_train, y_train)  # Learn parameters from train only
X_train_enc = encoder.transform(X_train)
X_test_enc = encoder.transform(X_test)  # Apply learned parameters to test
```

Or better yet, use sklearn pipelines which handle this automatically.

**3. Not Handling Unseen Categories at Test Time**

Training data has cities {NYC, LA, Chicago}. Test data has Miami. Without proper handling, encoding will fail:

```python
encoder = OneHotEncoder()
encoder.fit(X_train[['city']])  # Learns: NYC, LA, Chicago
encoder.transform(X_test[['city']])  # ERROR if Miami is in test set!
```

**Why it happens:** In development, the same dataset might be split, so train and test have the same categories. But in production, new categories emerge—new product types, new customer segments, new regions.

**What to do instead:** Always use `handle_unknown='ignore'` for OneHotEncoder:

```python
encoder = OneHotEncoder(handle_unknown='ignore')
```

Now unseen categories get encoded as all-zeros (representing "none of the known categories"). The model can still make predictions, treating the unseen category as neutral. For high-stakes applications, consider reserving an "Other" category during training to explicitly teach the model how to handle unknown inputs.

## Practice

**Practice 1**

Analyze movie data. Create a DataFrame with the following:

| Title    | Genre  | Rating |
|----------|--------|--------|
| Movie A  | Action | PG-13  |
| Movie B  | Comedy | R      |
| Movie C  | Action | PG     |
| Movie D  | Drama  | R      |
| Movie E  | Comedy | PG-13  |

1. Create this DataFrame in pandas
2. Use `pd.get_dummies()` to one-hot encode both categorical columns (Genre and Rating)
3. Print the shape before and after encoding—how many columns were created?
4. Try encoding again with `drop_first=True`. How does the result change and why?
5. Which approach (with or without `drop_first`) would be used for a linear regression model? Explain the reasoning.

**Practice 2**

Load a dataset with mixed categorical types (use `seaborn.load_dataset('titanic')` or create synthetic data with the following features):
- `sex`: nominal (male, female)
- `embarked`: nominal (C, Q, S - port of embarkation)
- `pclass`: ordinal (1, 2, 3 - passenger class)
- `age`: numeric
- `fare`: numeric
- `survived`: target (0, 1)

1. Explore the categorical columns. Check unique values and counts for each.
2. Handle missing values appropriately (fill with most frequent value for categorical, median for numeric).
3. For each categorical feature, decide and implement appropriate encoding:
   - One-hot encode nominal features with low cardinality
   - Use ordinal encoding for `pclass` (1=First, 2=Second, 3=Third)
4. Create two complete pipelines:
   - **Pipeline A**: All features one-hot encoded
   - **Pipeline B**: Mixed encoding (chosen strategy from step 3)
5. Train a `LogisticRegression` model with each pipeline. Which performs better on a test set?
6. Compare the number of features in each version. What's the trade-off?
7. **Bonus**: Try repeating step 5 with `RandomForestClassifier`. Are different results seen? Why might tree-based models be less sensitive to encoding choice?

**Practice 3**

Build a house price prediction model with a high-cardinality categorical feature: `neighborhood` (50+ unique values).

1. Load California Housing dataset (`fetch_california_housing()`) or Ames Housing dataset
2. Create a synthetic `neighborhood` feature by binning latitude/longitude into ~50-100 categories
3. Implement three encoding strategies from scratch (don't just use libraries):
   - **Approach A**: One-hot encoding (baseline)
   - **Approach B**: Frequency encoding (replace with category frequency)
   - **Approach C**: Target encoding with leave-one-out to prevent leakage:
     - For each sample, compute the mean target value for that neighborhood *excluding* the current sample's target
     - This prevents the encoding from "seeing" information about what it's trying to predict
4. For each approach:
   - Train a `Ridge` regression model
   - Evaluate on test set (R² score)
   - Measure training time
   - Count number of features after encoding
5. Create a comparison table or visualization showing (features, time, R²) for each approach
6. Reflection questions:
   - Why does naive target encoding (without leave-one-out) often show suspiciously high training performance but poor test performance?
   - What is "target leakage" and how does leave-one-out encoding address it?
   - In what scenarios would one-hot encoding still be preferred despite high cardinality?

## Solutions

**Solution 1**

```python
import pandas as pd

# Step 1: Create the DataFrame
df_movies = pd.DataFrame({
    'Title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Genre': ['Action', 'Comedy', 'Action', 'Drama', 'Comedy'],
    'Rating': ['PG-13', 'R', 'PG', 'R', 'PG-13']
})

print("Original DataFrame:")
print(df_movies)
print(f"Shape: {df_movies.shape}")

# Step 2: One-hot encode without drop_first
df_encoded = pd.get_dummies(df_movies, columns=['Genre', 'Rating'], drop_first=False)
print("\nOne-hot encoded (all categories):")
print(df_encoded)
print(f"Shape: {df_encoded.shape}")

# Step 3: Count columns created
original_cols = 3
encoded_cols = df_encoded.shape[1]
print(f"\nColumns created: {encoded_cols - original_cols}")
print("Genre had 3 unique values → 3 columns")
print("Rating had 3 unique values → 3 columns")

# Step 4: Encode with drop_first=True
df_encoded_drop = pd.get_dummies(df_movies, columns=['Genre', 'Rating'], drop_first=True)
print("\nOne-hot encoded (drop_first=True):")
print(df_encoded_drop)
print(f"Shape: {df_encoded_drop.shape}")
print("\nWith drop_first=True:")
print("Genre: 3 categories → 2 columns")
print("Rating: 3 categories → 2 columns")
print("This avoids the dummy variable trap (multicollinearity)")

# Step 5: Answer
print("\nFor linear regression:")
print("Use drop_first=True to avoid multicollinearity.")
print("If Genre_Action=0 and Genre_Comedy=0, it must be Drama.")
print("Including all three creates redundant information that confuses linear models.")
```

The approach uses `drop_first=True` for linear regression to avoid the dummy variable trap. When a model has an intercept term, keeping all K dummy variables creates perfect multicollinearity—the dropped category becomes the reference level, and other categories are interpreted relative to it.

**Solution 2**

```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Step 1: Explore categorical columns
print("Categorical column exploration:")
for col in ['sex', 'embarked', 'pclass']:
    print(f"\n{col}:")
    print(df[col].value_counts())
    print(f"Missing: {df[col].isnull().sum()}")

# Step 2: Prepare features
features = ['sex', 'embarked', 'pclass', 'age', 'fare']
X = df[features]
y = df['survived']

# Remove rows where target is missing
mask = y.notna()
X = X[mask]
y = y[mask]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3 & 4: Create pipelines

# Pipeline A: All one-hot encoded
numeric_features_a = ['age', 'fare']
categorical_features_a = ['sex', 'embarked', 'pclass']

preprocessor_a = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features_a),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ]), categorical_features_a)
    ])

pipeline_a = Pipeline([
    ('preprocessor', preprocessor_a),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Pipeline B: Mixed encoding (ordinal for pclass, one-hot for others)
numeric_features_b = ['age', 'fare']
nominal_features = ['sex', 'embarked']
ordinal_features = ['pclass']

preprocessor_b = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features_b),
        ('nom', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ]), nominal_features),
        ('ord', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(categories=[[1, 2, 3]]))
        ]), ordinal_features)
    ])

pipeline_b = Pipeline([
    ('preprocessor', preprocessor_b),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Step 5: Train and evaluate
pipeline_a.fit(X_train, y_train)
pipeline_b.fit(X_train, y_train)

print("\n" + "="*60)
print("Pipeline A (all one-hot):")
print(f"Train accuracy: {pipeline_a.score(X_train, y_train):.3f}")
print(f"Test accuracy: {pipeline_a.score(X_test, y_test):.3f}")

print("\nPipeline B (mixed encoding):")
print(f"Train accuracy: {pipeline_b.score(X_train, y_train):.3f}")
print(f"Test accuracy: {pipeline_b.score(X_test, y_test):.3f}")

# Step 6: Compare features
# Count features in each pipeline
print("\n" + "="*60)
print("Feature comparison:")
print("Pipeline A: ~8-9 features after encoding")
print("Pipeline B: ~6-7 features after encoding")
print("\nTrade-off:")
print("Pipeline B uses fewer features by preserving ordinal structure.")
print("This can reduce overfitting and improve interpretability.")
print("Both should perform similarly on this dataset.")

# Step 7: Bonus with RandomForest
pipeline_a_rf = Pipeline([
    ('preprocessor', preprocessor_a),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline_b_rf = Pipeline([
    ('preprocessor', preprocessor_b),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline_a_rf.fit(X_train, y_train)
pipeline_b_rf.fit(X_train, y_train)

print("\n" + "="*60)
print("Random Forest Results:")
print("Pipeline A (all one-hot):")
print(f"Test accuracy: {pipeline_a_rf.score(X_test, y_test):.3f}")

print("\nPipeline B (mixed encoding):")
print(f"Test accuracy: {pipeline_b_rf.score(X_test, y_test):.3f}")

print("\nWhy tree-based models are less sensitive:")
print("Trees split on thresholds (e.g., pclass <= 1.5).")
print("They don't use arithmetic on encoded values.")
print("Label encoding works fine: splits become category membership tests.")
```

Tree-based models are less sensitive to encoding choice because they split on thresholds rather than computing weighted sums. For trees, `pclass=2` isn't "twice" `pclass=1`—it's just a different category tested with splits like `pclass <= 1.5`.

**Solution 3**

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import time

# Step 1 & 2: Load data and create high-cardinality feature
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target

# Create neighborhood by binning lat/lon
df['neighborhood'] = (
    df['Latitude'].round(1).astype(str) + '_' +
    df['Longitude'].round(1).astype(str)
)

print(f"Total samples: {len(df)}")
print(f"Unique neighborhoods: {df['neighborhood'].nunique()}")

# Prepare features
X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'neighborhood']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Implement three encoding strategies

# Approach A: One-hot encoding
print("\n" + "="*60)
print("Approach A: One-Hot Encoding")
print("="*60)

start = time.time()
X_train_a = X_train.copy()
X_test_a = X_test.copy()

# One-hot encode neighborhood
train_dummies = pd.get_dummies(X_train_a['neighborhood'], prefix='neighborhood')
X_train_a = X_train_a.drop('neighborhood', axis=1).join(train_dummies)

# For test set, only include columns from training
test_dummies = pd.get_dummies(X_test_a['neighborhood'], prefix='neighborhood')
X_test_a = X_test_a.drop('neighborhood', axis=1).join(test_dummies)

# Align test columns with train columns
X_test_a = X_test_a.reindex(columns=X_train_a.columns, fill_value=0)

model_a = Ridge()
model_a.fit(X_train_a, y_train)
y_pred_a = model_a.predict(X_test_a)
r2_a = r2_score(y_test, y_pred_a)
time_a = time.time() - start
n_features_a = X_train_a.shape[1]

print(f"Features: {n_features_a}")
print(f"R² score: {r2_a:.4f}")
print(f"Time: {time_a:.2f}s")

# Approach B: Frequency encoding
print("\n" + "="*60)
print("Approach B: Frequency Encoding")
print("="*60)

start = time.time()
X_train_b = X_train.copy()
X_test_b = X_test.copy()

# Compute frequencies from training data only
freq_map = X_train_b['neighborhood'].value_counts(normalize=True).to_dict()
X_train_b['neighborhood'] = X_train_b['neighborhood'].map(freq_map)
X_test_b['neighborhood'] = X_test_b['neighborhood'].map(freq_map).fillna(0)

model_b = Ridge()
model_b.fit(X_train_b, y_train)
y_pred_b = model_b.predict(X_test_b)
r2_b = r2_score(y_test, y_pred_b)
time_b = time.time() - start
n_features_b = X_train_b.shape[1]

print(f"Features: {n_features_b}")
print(f"R² score: {r2_b:.4f}")
print(f"Time: {time_b:.2f}s")

# Approach C: Target encoding with leave-one-out
print("\n" + "="*60)
print("Approach C: Target Encoding (Leave-One-Out)")
print("="*60)

start = time.time()
X_train_c = X_train.copy()
X_test_c = X_test.copy()

# Compute leave-one-out encoding
def target_encode_loo(df, cat_col, target):
    global_mean = target.mean()
    encoded = []

    for idx in df.index:
        cat = df.loc[idx, cat_col]
        mask = (df[cat_col] == cat) & (df.index != idx)
        if mask.sum() > 0:
            encoded.append(target[mask].mean())
        else:
            encoded.append(global_mean)

    return encoded

# Apply leave-one-out on training
X_train_c['neighborhood'] = target_encode_loo(X_train_c, 'neighborhood', y_train)

# For test set, use simple mean from training (no leave-one-out needed)
target_map = X_train.groupby('neighborhood')['target'].agg('mean').to_dict()
global_mean = y_train.mean()
X_test_c['neighborhood'] = X_test_c['neighborhood'].map(target_map).fillna(global_mean)

model_c = Ridge()
model_c.fit(X_train_c, y_train)
y_pred_c = model_c.predict(X_test_c)
r2_c = r2_score(y_test, y_pred_c)
time_c = time.time() - start
n_features_c = X_train_c.shape[1]

print(f"Features: {n_features_c}")
print(f"R² score: {r2_c:.4f}")
print(f"Time: {time_c:.2f}s")

# Step 5: Comparison table
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)

comparison = pd.DataFrame({
    'Approach': ['One-Hot', 'Frequency', 'Target (LOO)'],
    'Features': [n_features_a, n_features_b, n_features_c],
    'R² Score': [r2_a, r2_b, r2_c],
    'Time (s)': [time_a, time_b, time_c]
})
print(comparison)

# Step 6: Reflection answers
print("\n" + "="*60)
print("REFLECTION ANSWERS")
print("="*60)

print("\n1. Why naive target encoding shows high train, poor test performance:")
print("   Without leave-one-out, each sample's encoding includes its own target.")
print("   The model memorizes these values instead of learning patterns.")
print("   This is direct target leakage causing severe overfitting.")

print("\n2. Target leakage and leave-one-out:")
print("   Target leakage: Using information about the target to create features.")
print("   Leave-one-out: Computes category mean excluding current sample.")
print("   This ensures the encoding doesn't 'see' the target it's predicting.")

print("\n3. When to still prefer one-hot encoding:")
print("   - Need model interpretability (clear coefficient per category)")
print("   - Categories have truly distinct patterns (not just different means)")
print("   - Cardinality is manageable (< 15-20 categories)")
print("   - Target encoding might create spurious correlations")
```

The comparison shows that target encoding typically achieves the best R² score with minimal features, but requires careful implementation to avoid leakage. One-hot encoding provides interpretability at the cost of dimensionality. Frequency encoding offers a middle ground—compact representation with reasonable performance, though it loses information about which specific neighborhood is which.

## Key Takeaways

- **Categorical features must be encoded as numbers for machine learning models**, but the encoding strategy must match the variable type: one-hot for nominal, ordinal encoding for ordinal, simple binary for two-level categories.

- **One-hot encoding creates K-1 binary columns for K categories** (with `drop_first=True`) to avoid multicollinearity, where each column indicates presence/absence of a category without imposing false numerical relationships.

- **Label encoding is only appropriate for ordinal variables or tree-based models**—using it on nominal features with linear models creates false mathematical relationships (Male=1, Female=2 implies Female is "twice" Male).

- **High-cardinality features** (50+ categories) require specialized strategies: frequency encoding, grouping rare categories, target encoding (with leave-one-out to prevent leakage), or feature hashing—one-hot encoding becomes impractical due to dimensionality explosion.

- **Always use pipelines and fit encoders on training data only** to prevent data leakage. Use `handle_unknown='ignore'` in production to gracefully handle categories in test/production data that weren't seen during training.

**Next:** Chapter 15 covers feature scaling and normalization, essential techniques for ensuring features with different units and ranges contribute appropriately to model learning.
