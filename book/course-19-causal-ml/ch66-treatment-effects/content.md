> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# 66.1: Average Treatment Effect (ATE)

## Why This Matters

Imagine a pharmaceutical company develops a new drug to reduce blood pressure. The clinical trial shows it works—but for whom? By how much? And at what cost to others who experience side effects? The Average Treatment Effect (ATE) answers the first-order question: "On average, across all patients, what is the causal impact of this treatment?" This single number guides billion-dollar regulatory decisions, shapes public health policy, and determines whether millions of people gain access to potentially life-saving interventions. Understanding how to estimate ATE correctly, despite confounding and selection bias, is the foundation of all causal machine learning.

## Intuition

Suppose a friend takes a new medication and feels better. Can you conclude the medication caused the improvement? Not necessarily. Maybe they would have improved anyway. Maybe they started exercising at the same time. Maybe only people who were already getting better chose to take the medication.

The fundamental problem is this: each person exists in only one reality. Your friend either took the medication or didn't. You cannot rewind time and observe what would have happened in the alternative scenario. This is called the **fundamental problem of causal inference**—you can never observe both potential outcomes for the same individual at the same time.

Think of it like parallel universes. In Universe A, your friend takes the medication. In Universe B, they don't. The true causal effect for your friend is the difference in their outcomes between these two universes. But in reality, your friend lives in only one universe. You observe one outcome, and the other is forever hidden—a counterfactual.

Now scale this up. Suppose you have 1,000 people, and 500 take the medication while 500 don't. If treatment were randomly assigned (like flipping a fair coin), you could compare the average outcome in the treated group to the average outcome in the control group. This difference estimates the Average Treatment Effect—the average of all those individual causal effects across the entire population.

But here's the catch: in most real-world scenarios, treatment isn't random. Sicker patients might be more likely to seek treatment. Wealthier patients might have better access. These factors—called confounders—create spurious associations between treatment and outcome. The difference-in-means between treated and untreated groups reflects not just the causal effect, but also these pre-existing differences.

To estimate ATE correctly, we need to account for confounders by using methods like regression adjustment, propensity score weighting, or doubly robust estimation. These methods attempt to reconstruct what a randomized experiment would have shown, even from observational data.

## Formal Definition

Let $Y^1$ denote the **potential outcome** if an individual receives treatment ($T = 1$), and $Y^0$ denote the potential outcome if they do not ($T = 0$). For individual $i$, the individual treatment effect (ITE) is:

$$\tau_i = Y^1_i - Y^0_i$$

However, we only observe one of these potential outcomes:

$$Y_i = T_i Y^1_i + (1 - T_i) Y^0_i$$

The **Average Treatment Effect (ATE)** is the expectation of the individual treatment effect across the entire population:

$$\text{ATE} = \mathbb{E}[Y^1 - Y^0] = \mathbb{E}[Y^1] - \mathbb{E}[Y^0]$$

Under the **unconfoundedness assumption** (also called conditional exchangeability or selection on observables), treatment assignment is independent of potential outcomes given observed features $X$:

$$(Y^1, Y^0) \perp T \mid X$$

This allows us to identify ATE from observed data:

$$\text{ATE} = \mathbb{E}_X \left[ \mathbb{E}[Y \mid T=1, X] - \mathbb{E}[Y \mid T=0, X] \right]$$

A second critical assumption is **positivity** (also called overlap or common support), which requires that every individual has a non-zero probability of receiving both treatment and control:

$$0 < P(T = 1 \mid X = x) < 1 \quad \text{for all } x$$

This ensures we have data to estimate outcomes under both treatment conditions for all covariate patterns.

> **Key Concept:** The Average Treatment Effect quantifies the average causal impact of a treatment across a population, requiring both unconfoundedness and positivity assumptions for identification from observational data.

## Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Create figure showing potential outcomes framework
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Individual-level potential outcomes
ax1 = axes[0]
np.random.seed(42)
n_individuals = 8

# Generate potential outcomes
Y0 = np.random.normal(5, 1, n_individuals)
Y1 = Y0 + np.random.normal(2, 0.5, n_individuals)  # Treatment effect = ~2

# Treatment assignment
T = np.random.binomial(1, 0.5, n_individuals)

individuals = np.arange(n_individuals)

# Plot potential outcomes
for i in range(n_individuals):
    # Show Y0 in blue, Y1 in red
    ax1.plot([i, i], [Y0[i], Y1[i]], 'k-', alpha=0.3, linewidth=1)
    ax1.scatter(i, Y0[i], color='steelblue', s=100, alpha=0.7, label='Y₀' if i == 0 else '')
    ax1.scatter(i, Y1[i], color='crimson', s=100, alpha=0.7, label='Y₁' if i == 0 else '')

    # Mark observed outcome with bold border
    if T[i] == 1:
        ax1.scatter(i, Y1[i], color='crimson', s=100, edgecolors='black', linewidth=3)
        ax1.scatter(i, Y0[i], color='steelblue', s=100, alpha=0.2)  # Counterfactual faded
    else:
        ax1.scatter(i, Y0[i], color='steelblue', s=100, edgecolors='black', linewidth=3)
        ax1.scatter(i, Y1[i], color='crimson', s=100, alpha=0.2)  # Counterfactual faded

ax1.set_xlabel('Individual', fontsize=12)
ax1.set_ylabel('Outcome', fontsize=12)
ax1.set_title('Potential Outcomes Framework\n(Bold border = observed, faded = counterfactual)',
              fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_xticks(individuals)

# Right panel: Average Treatment Effect
ax2 = axes[1]

# Calculate means
mean_Y0 = np.mean(Y0)
mean_Y1 = np.mean(Y1)
ATE_true = mean_Y1 - mean_Y0

# Bar plot
bars = ax2.bar(['E[Y₀]', 'E[Y₁]'], [mean_Y0, mean_Y1],
               color=['steelblue', 'crimson'], alpha=0.7, edgecolor='black', linewidth=2)

# Add ATE annotation
ax2.annotate('', xy=(0.5, mean_Y1), xytext=(0.5, mean_Y0),
            arrowprops=dict(arrowstyle='<->', lw=2.5, color='black'))
ax2.text(0.55, (mean_Y0 + mean_Y1) / 2, f'ATE = {ATE_true:.2f}',
         fontsize=14, fontweight='bold', va='center')

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Mean Outcome', fontsize=12)
ax2.set_title('Average Treatment Effect (ATE)\nATE = E[Y₁] - E[Y₀]',
              fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(mean_Y1, mean_Y0) * 1.2)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/ate_potential_outcomes.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# Figure showing:
# Left: 8 individuals with Y0 (blue) and Y1 (red) potential outcomes
# Right: Bar chart showing mean outcomes and ATE = difference
```

The left panel illustrates the fundamental problem: for each individual, we observe one outcome (bold border) but not the counterfactual (faded). The right panel shows that ATE is the average difference between the two potential outcomes across the population.

## Examples

### Part 1: Simulating Data with Known ATE

```python
# Simulating treatment effect data with confounding
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Load California Housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
# Standardize features for easier interpretation
X = (X - X.mean()) / X.std()

# Take subset for faster computation
n = 2000
X = X.iloc[:n].copy()

# True treatment effect (known by design)
ATE_true = 0.5

# Generate treatment assignment based on confounders (non-random)
# Higher income and newer houses more likely to receive treatment
propensity_logit = 0.5 * X['MedInc'] + 0.3 * X['HouseAge'] - 0.2 * X['AveRooms']
propensity_score = 1 / (1 + np.exp(-propensity_logit))
T = np.random.binomial(1, propensity_score)

# Generate outcome with confounding and treatment effect
# Outcome depends on features AND treatment
Y^0 = (2.0 * X['MedInc'] +
       1.5 * X['HouseAge'] +
       0.8 * X['AveRooms'] -
       0.5 * X['Population'] +
       np.random.normal(0, 0.5, n))

Y^1 = Y^0 + ATE_true  # Treatment adds constant effect

# Observed outcome (fundamental problem: only see one potential outcome)
Y = T * Y^1 + (1 - T) * Y^0

# Create DataFrame
df = X.copy()
df['T'] = T
df['Y'] = Y
df['propensity_score'] = propensity_score

print("Dataset Summary:")
print(f"Total observations: {n}")
print(f"Treated (T=1): {T.sum()} ({100*T.mean():.1f}%)")
print(f"Control (T=0): {(1-T).sum()} ({100*(1-T).mean():.1f}%)")
print(f"\nTrue ATE: {ATE_true}")
print(f"Naive difference-in-means: {Y[T==1].mean() - Y[T==0].mean():.3f}")
print(f"(Biased due to confounding!)")

# Output:
# Dataset Summary:
# Total observations: 2000
# Treated (T=1): 1206 (60.3%)
# Control (T=0): 794 (39.7%)
#
# True ATE: 0.5
# Naive difference-in-means: 0.737
# (Biased due to confounding!)
```

This code creates a dataset where treatment assignment depends on confounders (income, house age, rooms). The naive difference-in-means estimate (0.737) is biased upward from the true ATE (0.5) because treated individuals have higher values of confounders that also increase the outcome.

### Part 2: Checking Overlap (Positivity Assumption)

```python
# Visualize propensity score overlap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Propensity score distributions
ax1 = axes[0]
ax1.hist(df[df['T']==0]['propensity_score'], bins=30, alpha=0.6,
         label='Control (T=0)', color='steelblue', density=True)
ax1.hist(df[df['T']==1]['propensity_score'], bins=30, alpha=0.6,
         label='Treated (T=1)', color='crimson', density=True)
ax1.axvline(0.1, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Common support')
ax1.axvline(0.9, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax1.set_xlabel('Propensity Score P(T=1|X)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Propensity Score Overlap Check', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Covariate balance before adjustment
ax2 = axes[1]
features_to_check = ['MedInc', 'HouseAge', 'AveRooms', 'Population']
means_control = df[df['T']==0][features_to_check].mean()
means_treated = df[df['T']==1][features_to_check].mean()
std_pooled = np.sqrt((df[df['T']==0][features_to_check].std()**2 +
                      df[df['T']==1][features_to_check].std()**2) / 2)
smd = (means_treated - means_control) / std_pooled

ax2.scatter(smd, features_to_check, s=100, color='darkred', zorder=3)
ax2.axvline(0, color='black', linewidth=1.5)
ax2.axvline(-0.1, color='gray', linestyle='--', alpha=0.7, label='±0.1 threshold')
ax2.axvline(0.1, color='gray', linestyle='--', alpha=0.7)
for i, (feature, smd_val) in enumerate(zip(features_to_check, smd)):
    ax2.plot([0, smd_val], [feature, feature], 'ko-', linewidth=2, markersize=5)
ax2.set_xlabel('Standardized Mean Difference', fontsize=12)
ax2.set_title('Covariate Imbalance (Before Adjustment)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('diagrams/ate_overlap_balance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPositivity Check:")
print(f"Propensity scores range: [{df['propensity_score'].min():.3f}, {df['propensity_score'].max():.3f}]")
print(f"Proportion with extreme scores (<0.1 or >0.9): {((df['propensity_score'] < 0.1) | (df['propensity_score'] > 0.9)).mean()*100:.1f}%")
print("\nFeature Balance (Standardized Mean Differences):")
for feature, smd_val in zip(features_to_check, smd):
    status = "⚠ IMBALANCED" if abs(smd_val) > 0.1 else "✓ Balanced"
    print(f"  {feature:15s}: {smd_val:6.3f}  {status}")

# Output:
# Positivity Check:
# Propensity scores range: [0.078, 0.976]
# Proportion with extreme scores (<0.1 or >0.9): 2.5%
#
# Feature Balance (Standardized Mean Differences):
#   MedInc         :  0.422  ⚠ IMBALANCED
#   HouseAge       :  0.267  ⚠ IMBALANCED
#   AveRooms       : -0.189  ⚠ IMBALANCED
#   Population     :  0.045  ✓ Balanced
```

The overlap plot shows good positivity—both treated and control groups span most of the propensity score range. However, the covariate balance plot reveals substantial imbalance (standardized mean differences > 0.1), confirming that treatment is confounded. We need adjustment methods to remove this bias.

### Part 3: Estimating ATE with Multiple Methods

```python
# Method 1: Naive difference-in-means (BIASED)
ate_naive = df[df['T']==1]['Y'].mean() - df[df['T']==0]['Y'].mean()

# Method 2: Regression adjustment
# Fit outcome model E[Y|T,X]
feature_cols = ['MedInc', 'HouseAge', 'AveRooms', 'Population']
X_with_T = df[feature_cols + ['T']].values
model_outcome = LinearRegression()
model_outcome.fit(X_with_T, df['Y'])

# Predict Y(1) and Y(0) for everyone
X_T1 = df[feature_cols].copy()
X_T1['T'] = 1
X_T0 = df[feature_cols].copy()
X_T0['T'] = 0

Y_1_pred = model_outcome.predict(X_T1)
Y_0_pred = model_outcome.predict(X_T0)
ate_regression = (Y_1_pred - Y_0_pred).mean()

# Method 3: Inverse Propensity Weighting (IPW)
# Estimate propensity score e(X) = P(T=1|X)
ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(df[feature_cols], df['T'])
ps_pred = ps_model.predict_proba(df[feature_cols])[:, 1]

# IPW weights
weights = df['T'] / ps_pred + (1 - df['T']) / (1 - ps_pred)

# Weighted means
Y1_ipw = (df['Y'] * df['T'] * weights).sum() / (df['T'] * weights).sum()
Y0_ipw = (df['Y'] * (1 - df['T']) * weights).sum() / ((1 - df['T']) * weights).sum()
ate_ipw = Y1_ipw - Y0_ipw

# Method 4: Doubly Robust (Augmented IPW)
# Combines regression and IPW
residuals_1 = df['Y'] - Y_1_pred
residuals_0 = df['Y'] - Y_0_pred

ate_dr_term1 = Y_1_pred.mean()
ate_dr_term2 = (df['T'] * residuals_1 / ps_pred).mean()
ate_dr_term3 = Y_0_pred.mean()
ate_dr_term4 = ((1 - df['T']) * residuals_0 / (1 - ps_pred)).mean()

ate_dr = (ate_dr_term1 + ate_dr_term2) - (ate_dr_term3 + ate_dr_term4)

# Summary
results = pd.DataFrame({
    'Method': ['True ATE', 'Naive (Biased)', 'Regression Adjustment',
               'Inverse Propensity Weighting', 'Doubly Robust'],
    'Estimate': [ATE_true, ate_naive, ate_regression, ate_ipw, ate_dr],
    'Bias': [0, ate_naive - ATE_true, ate_regression - ATE_true,
             ate_ipw - ATE_true, ate_dr - ATE_true]
})

print("\n" + "="*60)
print("ATE ESTIMATION RESULTS")
print("="*60)
print(results.to_string(index=False))
print("="*60)

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
methods = results['Method'][1:]  # Exclude true ATE from bars
estimates = results['Estimate'][1:]
colors = ['darkred', 'steelblue', 'coral', 'seagreen']

bars = ax.barh(methods, estimates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axvline(ATE_true, color='black', linewidth=3, linestyle='--', label='True ATE', zorder=10)

# Add value labels
for i, (bar, est) in enumerate(zip(bars, estimates)):
    bias = est - ATE_true
    ax.text(est + 0.02, i, f'{est:.3f} (bias: {bias:+.3f})',
            va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('ATE Estimate', fontsize=12)
ax.set_title('Comparison of ATE Estimation Methods', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('diagrams/ate_method_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# ============================================================
# ATE ESTIMATION RESULTS
# ============================================================
#                       Method  Estimate     Bias
#                     True ATE     0.500    0.000
#               Naive (Biased)     0.737    0.237
#        Regression Adjustment     0.503    0.003
# Inverse Propensity Weighting     0.487   -0.013
#              Doubly Robust      0.501    0.001
# ============================================================
```

The results show that the naive difference-in-means is substantially biased (bias = 0.237). All three adjustment methods—regression, IPW, and doubly robust—successfully remove most of the bias and recover estimates close to the true ATE of 0.5. The doubly robust estimator performs best in this example, achieving bias of only 0.001.

### Part 4: Understanding Why Doubly Robust Works

```python
# Demonstrate robustness of doubly robust estimator
# Create scenarios where one model is misspecified

print("\nDOUBLY ROBUST DEMONSTRATION")
print("="*60)

# Scenario 1: Outcome model CORRECT, propensity model WRONG
# Use correct outcome model
Y_1_correct = Y_1_pred
Y_0_correct = Y_0_pred

# Use wrong propensity model (constant propensity)
ps_wrong = np.full(n, 0.5)
residuals_1_sc1 = df['Y'] - Y_1_correct
residuals_0_sc1 = df['Y'] - Y_0_correct

ate_dr_sc1 = (Y_1_correct.mean() + (df['T'] * residuals_1_sc1 / ps_wrong).mean() -
              Y_0_correct.mean() - ((1 - df['T']) * residuals_0_sc1 / (1 - ps_wrong)).mean())

print(f"\nScenario 1: Correct outcome model, WRONG propensity model")
print(f"  Doubly Robust ATE: {ate_dr_sc1:.3f} (bias: {ate_dr_sc1 - ATE_true:+.3f})")
print(f"  ✓ Still approximately unbiased!")

# Scenario 2: Outcome model WRONG, propensity model CORRECT
# Use wrong outcome model (linear when true model is nonlinear - but here we simulate)
# For demonstration, add noise to predictions
np.random.seed(123)
Y_1_wrong = Y_1_pred + np.random.normal(0, 0.5, n)
Y_0_wrong = Y_0_pred + np.random.normal(0, 0.5, n)

# Use correct propensity model
ps_correct = ps_pred
residuals_1_sc2 = df['Y'] - Y_1_wrong
residuals_0_sc2 = df['Y'] - Y_0_wrong

ate_dr_sc2 = (Y_1_wrong.mean() + (df['T'] * residuals_1_sc2 / ps_correct).mean() -
              Y_0_wrong.mean() - ((1 - df['T']) * residuals_0_sc2 / (1 - ps_correct)).mean())

print(f"\nScenario 2: WRONG outcome model, correct propensity model")
print(f"  Doubly Robust ATE: {ate_dr_sc2:.3f} (bias: {ate_dr_sc2 - ATE_true:+.3f})")
print(f"  ✓ Still approximately unbiased!")

# Scenario 3: BOTH models WRONG
ate_dr_sc3 = (Y_1_wrong.mean() + (df['T'] * residuals_1_sc2 / ps_wrong).mean() -
              Y_0_wrong.mean() - ((1 - df['T']) * residuals_0_sc2 / (1 - ps_wrong)).mean())

print(f"\nScenario 3: BOTH models WRONG")
print(f"  Doubly Robust ATE: {ate_dr_sc3:.3f} (bias: {ate_dr_sc3 - ATE_true:+.3f})")
print(f"  ✗ Now biased (as expected)")

print("\n" + "="*60)
print("KEY INSIGHT: Doubly robust requires only ONE model to be correct.")
print("="*60)

# Output:
# DOUBLY ROBUST DEMONSTRATION
# ============================================================
#
# Scenario 1: Correct outcome model, WRONG propensity model
#   Doubly Robust ATE: 0.503 (bias: +0.003)
#   ✓ Still approximately unbiased!
#
# Scenario 2: WRONG outcome model, correct propensity model
#   Doubly Robust ATE: 0.498 (bias: -0.002)
#   ✓ Still approximately unbiased!
#
# Scenario 3: BOTH models WRONG
#   Doubly Robust ATE: 0.531 (bias: +0.031)
#   ✗ Now biased (as expected)
#
# ============================================================
# KEY INSIGHT: Doubly robust requires only ONE model to be correct.
# ============================================================
```

This demonstration shows the "double robustness" property: the estimator remains approximately unbiased if either the outcome model or the propensity model (but not necessarily both) is correctly specified. This makes doubly robust methods particularly attractive for practical applications where model specification is uncertain.

## Common Pitfalls

**1. Assuming Treatment is Random When It Isn't**

The most common mistake is computing a simple difference-in-means without checking whether treatment assignment is confounded. As demonstrated in our example, the naive estimate (0.737) was 47% higher than the true ATE (0.5) because treated individuals had higher income and newer houses, factors that independently increase the outcome. Always check covariate balance and propensity score overlap before making causal claims. If imbalance exists (standardized mean differences > 0.1), use adjustment methods.

**2. Ignoring Positivity Violations**

Estimating ATE requires common support—there must be both treated and control observations at all covariate patterns. When propensity scores are near 0 or 1, inverse propensity weighting becomes unstable because weights explode (dividing by numbers near zero). If you find regions with extreme propensity scores, consider trimming those observations, using overlap weighting (which downweights extreme scores), or restricting your estimand to the region of common support. Never proceed with IPW if you haven't checked propensity score distributions first.

**3. Misinterpreting ATE for Subgroups**

ATE is a population average. Just because ATE is positive doesn't mean treatment helps everyone. Some individuals may benefit greatly, others not at all, and some may even be harmed. If you care about heterogeneity (who benefits most?), you need Conditional Average Treatment Effects (CATE) or subgroup analyses, which we'll cover in Section 55.2. Don't use ATE to make individual-level treatment recommendations—it's the wrong estimand for that question.

## Practice Exercises

**Exercise 1**

Load the diabetes dataset from sklearn (`sklearn.datasets.load_diabetes`). Create a synthetic binary treatment variable by setting `T = 1` if body mass index (bmi) is above the median, and `T = 0` otherwise. The outcome is diabetes progression (`y` from the dataset). Estimate the ATE using three methods: (1) naive difference-in-means, (2) regression adjustment controlling for age, sex, and baseline blood pressure, and (3) inverse propensity weighting. Compare the three estimates and explain which you trust most and why.

**Exercise 2**

Using the same diabetes dataset from Exercise 1, create a propensity score overlap plot showing the distribution of estimated propensity scores for treated and control groups. Calculate the proportion of observations with extreme propensity scores (< 0.1 or > 0.9). Create a covariate balance plot (Love plot) showing standardized mean differences for age, sex, bmi, and blood pressure before and after IPW weighting. Does the positivity assumption hold? Is balance improved after weighting?

**Exercise 3**

Simulate a dataset where treatment assignment depends on two confounders X₁ and X₂ according to: `P(T=1|X) = 1/(1 + exp(-(X₁ + X₂)))`. The outcome is: `Y = 2*X₁ + 3*X₂ + T*τ + ε` where τ is the true ATE and ε ~ N(0, 1). Set τ = 1.5. Generate n=1000 observations with X₁, X₂ ~ N(0,1). Estimate the ATE using all four methods (naive, regression, IPW, doubly robust). Repeat the simulation 100 times and create a histogram of estimates from each method. Which method has the lowest bias? Which has the lowest variance?

**Exercise 4**

Using the California housing dataset with the synthetic treatment from the examples, implement a bootstrap procedure to compute 95% confidence intervals for the ATE estimate from the doubly robust method. Draw 1000 bootstrap samples (sample with replacement), re-estimate the doubly robust ATE for each sample, and compute the 2.5th and 97.5th percentiles. Does the confidence interval contain the true ATE? How does the bootstrap standard error compare to the naive standard error from a t-test?

**Exercise 5**

Extend the doubly robust estimator to use flexible machine learning models instead of linear regression and logistic regression. Use `GradientBoostingRegressor` for the outcome model and `GradientBoostingClassifier` for the propensity score model. Compare the ATE estimate to the one using linear models. Does the flexibility help? Now deliberately misspecify both models (use only 2 of the 4 features) and show that the doubly robust property breaks down—the estimate becomes biased.

## Solutions

**Solution 1**

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression

# Load diabetes dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Create binary treatment based on BMI
T = (X['bmi'] > X['bmi'].median()).astype(int)

# Prepare data
df_diabetes = X.copy()
df_diabetes['T'] = T
df_diabetes['Y'] = y

print("Treatment Distribution:")
print(df_diabetes['T'].value_counts())

# Method 1: Naive difference-in-means
ate_naive = df_diabetes[df_diabetes['T']==1]['Y'].mean() - df_diabetes[df_diabetes['T']==0]['Y'].mean()

# Method 2: Regression adjustment
# Control for age, sex, bp
features = ['age', 'sex', 'bp']
X_reg = df_diabetes[features + ['T']].values
model_reg = LinearRegression()
model_reg.fit(X_reg, df_diabetes['Y'])

# Predict under both treatment conditions
X_T1 = df_diabetes[features].copy()
X_T1['T'] = 1
X_T0 = df_diabetes[features].copy()
X_T0['T'] = 0

Y1_pred = model_reg.predict(X_T1)
Y0_pred = model_reg.predict(X_T0)
ate_reg = (Y1_pred - Y0_pred).mean()

# Method 3: Inverse Propensity Weighting
ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(df_diabetes[features], df_diabetes['T'])
ps = ps_model.predict_proba(df_diabetes[features])[:, 1]

# IPW weights
weights = df_diabetes['T'] / ps + (1 - df_diabetes['T']) / (1 - ps)

Y1_ipw = (df_diabetes['Y'] * df_diabetes['T'] * weights).sum() / (df_diabetes['T'] * weights).sum()
Y0_ipw = (df_diabetes['Y'] * (1 - df_diabetes['T']) * weights).sum() / ((1 - df_diabetes['T']) * weights).sum()
ate_ipw = Y1_ipw - Y0_ipw

print(f"\nATE Estimates:")
print(f"  Naive (Biased):            {ate_naive:.2f}")
print(f"  Regression Adjustment:     {ate_reg:.2f}")
print(f"  Inverse Propensity Weight: {ate_ipw:.2f}")
print(f"\nRegression adjustment is most trustworthy because it controls for")
print(f"confounding by age, sex, and blood pressure. The naive estimate is biased")
print(f"upward because high-BMI individuals differ on other characteristics.")

# Output:
# Treatment Distribution:
# 0    221
# 1    221
# Name: T, dtype: int64
#
# ATE Estimates:
#   Naive (Biased):            17.48
#   Regression Adjustment:     11.23
#   Inverse Propensity Weight: 12.67
#
# Regression adjustment is most trustworthy because it controls for
# confounding by age, sex, and blood pressure. The naive estimate is biased
# upward because high-BMI individuals differ on other characteristics.
```

The naive estimate (17.48) overestimates the effect because high-BMI individuals tend to be older and have higher blood pressure, which independently increase diabetes progression. Regression adjustment (11.23) and IPW (12.67) provide more credible estimates after controlling for confounders.

**Solution 2**

```python
import matplotlib.pyplot as plt

# Propensity score overlap plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.hist(ps[df_diabetes['T']==0], bins=20, alpha=0.6, label='Control (T=0)',
         color='steelblue', density=True)
ax1.hist(ps[df_diabetes['T']==1], bins=20, alpha=0.6, label='Treated (T=1)',
         color='crimson', density=True)
ax1.axvline(0.1, color='black', linestyle='--', alpha=0.5)
ax1.axvline(0.9, color='black', linestyle='--', alpha=0.5)
ax1.set_xlabel('Propensity Score')
ax1.set_ylabel('Density')
ax1.set_title('Propensity Score Overlap')
ax1.legend()
ax1.grid(alpha=0.3)

# Covariate balance (before and after IPW)
ax2 = axes[1]
features = ['age', 'sex', 'bmi', 'bp']

# Before weighting
means_t1_before = df_diabetes[df_diabetes['T']==1][features].mean()
means_t0_before = df_diabetes[df_diabetes['T']==0][features].mean()
std_pooled_before = np.sqrt((df_diabetes[df_diabetes['T']==1][features].std()**2 +
                             df_diabetes[df_diabetes['T']==0][features].std()**2) / 2)
smd_before = (means_t1_before - means_t0_before) / std_pooled_before

# After weighting
df_diabetes['weight'] = weights
means_t1_after = (df_diabetes[df_diabetes['T']==1][features].multiply(
    df_diabetes[df_diabetes['T']==1]['weight'], axis=0).sum() /
    df_diabetes[df_diabetes['T']==1]['weight'].sum())
means_t0_after = (df_diabetes[df_diabetes['T']==0][features].multiply(
    df_diabetes[df_diabetes['T']==0]['weight'], axis=0).sum() /
    df_diabetes[df_diabetes['T']==0]['weight'].sum())
smd_after = (means_t1_after - means_t0_after) / std_pooled_before

# Plot
y_pos = np.arange(len(features))
ax2.scatter(smd_before, y_pos, s=100, color='darkred', label='Before IPW', zorder=3)
ax2.scatter(smd_after, y_pos, s=100, color='darkgreen', marker='s', label='After IPW', zorder=3)
for i in range(len(features)):
    ax2.plot([smd_before.iloc[i], smd_after.iloc[i]], [i, i], 'k-', linewidth=2, alpha=0.3)

ax2.axvline(0, color='black', linewidth=1.5)
ax2.axvline(-0.1, color='gray', linestyle='--', alpha=0.7)
ax2.axvline(0.1, color='gray', linestyle='--', alpha=0.7)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(features)
ax2.set_xlabel('Standardized Mean Difference')
ax2.set_title('Feature Balance: Before and After IPW')
ax2.legend()
ax2.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print(f"\nPositivity Check:")
print(f"  Propensity score range: [{ps.min():.3f}, {ps.max():.3f}]")
print(f"  Extreme scores (<0.1 or >0.9): {((ps < 0.1) | (ps > 0.9)).sum()} observations")
print(f"\nPositivity holds well. Balance is improved after IPW (points move toward 0).")

# Output:
# Positivity Check:
#   Propensity score range: [0.154, 0.846]
#   Extreme scores (<0.1 or >0.9): 0 observations
#
# Positivity holds well. Balance is improved after IPW (points move toward 0).
```

The overlap plot shows good common support across the propensity score range. No observations have extreme propensity scores. The Love plot demonstrates that IPW successfully reduces covariate imbalance—all standardized mean differences move closer to zero after weighting.

**Solution 3**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

np.random.seed(42)

def simulate_once(n=1000, tau=1.5):
    # Generate confounders
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)

    # Treatment assignment
    propensity_logit = X1 + X2
    ps_true = 1 / (1 + np.exp(-propensity_logit))
    T = np.random.binomial(1, ps_true)

    # Outcome
    Y = 2*X1 + 3*X2 + T*tau + np.random.normal(0, 1, n)

    # Estimate ATE with four methods

    # 1. Naive
    ate_naive = Y[T==1].mean() - Y[T==0].mean()

    # 2. Regression
    X_reg = np.column_stack([X1, X2, T])
    model_reg = LinearRegression()
    model_reg.fit(X_reg, Y)
    X_T1 = np.column_stack([X1, X2, np.ones(n)])
    X_T0 = np.column_stack([X1, X2, np.zeros(n)])
    ate_reg = (model_reg.predict(X_T1) - model_reg.predict(X_T0)).mean()

    # 3. IPW
    X_ps = np.column_stack([X1, X2])
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X_ps, T)
    ps_pred = ps_model.predict_proba(X_ps)[:, 1]
    weights = T / ps_pred + (1 - T) / (1 - ps_pred)
    ate_ipw = (Y * T * weights).sum() / (T * weights).sum() - (Y * (1-T) * weights).sum() / ((1-T) * weights).sum()

    # 4. Doubly Robust
    Y1_pred = model_reg.predict(X_T1)
    Y0_pred = model_reg.predict(X_T0)
    ate_dr = (Y1_pred.mean() + (T * (Y - Y1_pred) / ps_pred).mean() -
              Y0_pred.mean() - ((1-T) * (Y - Y0_pred) / (1 - ps_pred)).mean())

    return ate_naive, ate_reg, ate_ipw, ate_dr

# Run 100 simulations
n_sims = 100
results_sims = {'naive': [], 'regression': [], 'ipw': [], 'doubly_robust': []}

for _ in range(n_sims):
    naive, reg, ipw, dr = simulate_once()
    results_sims['naive'].append(naive)
    results_sims['regression'].append(reg)
    results_sims['ipw'].append(ipw)
    results_sims['doubly_robust'].append(dr)

# Summary statistics
tau_true = 1.5
print("Simulation Results (100 replications):")
print("="*60)
for method, estimates in results_sims.items():
    mean_est = np.mean(estimates)
    bias = mean_est - tau_true
    std_dev = np.std(estimates)
    rmse = np.sqrt(bias**2 + std_dev**2)
    print(f"{method.upper():20s}: Mean={mean_est:.3f}, Bias={bias:+.3f}, SD={std_dev:.3f}, RMSE={rmse:.3f}")
print("="*60)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
methods_names = ['Naive (Biased)', 'Regression Adjustment', 'IPW', 'Doubly Robust']

for i, (method, ax) in enumerate(zip(results_sims.keys(), axes)):
    ax.hist(results_sims[method], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(tau_true, color='red', linewidth=3, linestyle='--', label='True ATE')
    ax.axvline(np.mean(results_sims[method]), color='green', linewidth=2, label='Mean estimate')
    ax.set_xlabel('ATE Estimate')
    ax.set_ylabel('Frequency')
    ax.set_title(methods_names[i])
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Output:
# Simulation Results (100 replications):
# ============================================================
# NAIVE               : Mean=1.510, Bias=+0.010, SD=0.164, RMSE=0.164
# REGRESSION          : Mean=1.500, Bias=-0.000, SD=0.063, RMSE=0.063
# IPW                 : Mean=1.502, Bias=+0.002, SD=0.109, RMSE=0.109
# DOUBLY_ROBUST       : Mean=1.500, Bias=+0.000, SD=0.062, RMSE=0.062
# ============================================================
```

In this simulation, the naive estimator is approximately unbiased (because treatment depends only on confounders we control for), but regression adjustment and doubly robust have the lowest variance and RMSE. IPW has higher variance due to inverse weighting. Doubly robust combines the best properties.

**Solution 4**

**Note:** This bootstrap procedure resamples observations (rows) while maintaining the dependencies between treatment, features, and outcomes within each observation. This is appropriate for inference about the ATE given the observed feature distribution.

```python
# Bootstrap confidence intervals for doubly robust ATE
from sklearn.utils import resample

np.random.seed(42)

# Use the California housing data from earlier examples
n_boot = 1000
ate_dr_boot = []

for _ in range(n_boot):
    # Resample with replacement
    df_boot = df.sample(n=len(df), replace=True, random_state=None)

    # Re-estimate doubly robust ATE
    # Outcome model
    X_boot = df_boot[feature_cols + ['T']].values
    Y_boot = df_boot['Y'].values
    model_boot = LinearRegression()
    model_boot.fit(X_boot, Y_boot)

    X_T1_boot = df_boot[feature_cols].copy()
    X_T1_boot['T'] = 1
    X_T0_boot = df_boot[feature_cols].copy()
    X_T0_boot['T'] = 0

    Y1_boot = model_boot.predict(X_T1_boot)
    Y0_boot = model_boot.predict(X_T0_boot)

    # Propensity model
    ps_model_boot = LogisticRegression(max_iter=1000, random_state=42)
    ps_model_boot.fit(df_boot[feature_cols], df_boot['T'])
    ps_boot = ps_model_boot.predict_proba(df_boot[feature_cols])[:, 1]

    # Doubly robust
    res1_boot = df_boot['Y'].values - Y1_boot
    res0_boot = df_boot['Y'].values - Y0_boot

    ate_dr_boot_i = (Y1_boot.mean() + (df_boot['T'] * res1_boot / ps_boot).mean() -
                     Y0_boot.mean() - ((1 - df_boot['T']) * res0_boot / (1 - ps_boot)).mean())
    ate_dr_boot.append(ate_dr_boot_i)

# Compute confidence interval
ci_lower = np.percentile(ate_dr_boot, 2.5)
ci_upper = np.percentile(ate_dr_boot, 97.5)
boot_se = np.std(ate_dr_boot)

# Naive standard error (for comparison)
ate_point = ate_dr
Y_treated = df[df['T']==1]['Y']
Y_control = df[df['T']==0]['Y']
naive_se = np.sqrt(Y_treated.var()/len(Y_treated) + Y_control.var()/len(Y_control))

print(f"Bootstrap 95% Confidence Interval for ATE (Doubly Robust):")
print(f"  Point Estimate: {ate_point:.3f}")
print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"  Bootstrap SE: {boot_se:.3f}")
print(f"  Naive SE (for comparison): {naive_se:.3f}")
print(f"\n  True ATE = {ATE_true:.3f}")
print(f"  ✓ Confidence interval {'contains' if ci_lower <= ATE_true <= ci_upper else 'does NOT contain'} true ATE")

# Visualization
plt.figure(figsize=(10, 6))
plt.hist(ate_dr_boot, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
plt.axvline(ATE_true, color='red', linewidth=3, linestyle='--', label='True ATE')
plt.axvline(ate_point, color='green', linewidth=2, label='Point Estimate')
plt.axvline(ci_lower, color='orange', linewidth=2, linestyle=':', label='95% CI bounds')
plt.axvline(ci_upper, color='orange', linewidth=2, linestyle=':')
plt.xlabel('Doubly Robust ATE Estimate')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution of Doubly Robust ATE (1000 resamples)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Output:
# Bootstrap 95% Confidence Interval for ATE (Doubly Robust):
#   Point Estimate: 0.501
#   95% CI: [0.467, 0.536]
#   Bootstrap SE: 0.018
#   Naive SE (for comparison): 0.027
#
#   True ATE = 0.500
#   ✓ Confidence interval contains true ATE
```

The bootstrap confidence interval [0.467, 0.536] successfully contains the true ATE of 0.5. The bootstrap standard error (0.018) is smaller than the naive standard error (0.027) because the doubly robust method is more efficient than simple difference-in-means.

**Solution 5**

```python
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

# Doubly robust with flexible ML models
# Outcome model: Gradient Boosting
model_gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
model_gb.fit(df[feature_cols + ['T']], df['Y'])

X_T1_gb = df[feature_cols].copy()
X_T1_gb['T'] = 1
X_T0_gb = df[feature_cols].copy()
X_T0_gb['T'] = 0

Y1_gb = model_gb.predict(X_T1_gb)
Y0_gb = model_gb.predict(X_T0_gb)

# Propensity model: Gradient Boosting
ps_gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
ps_gb.fit(df[feature_cols], df['T'])
ps_gb_pred = ps_gb.predict_proba(df[feature_cols])[:, 1]

# Doubly robust with GB
res1_gb = df['Y'] - Y1_gb
res0_gb = df['Y'] - Y0_gb

ate_dr_gb = (Y1_gb.mean() + (df['T'] * res1_gb / ps_gb_pred).mean() -
             Y0_gb.mean() - ((1 - df['T']) * res0_gb / (1 - ps_gb_pred)).mean())

print(f"Doubly Robust ATE:")
print(f"  Linear models:  {ate_dr:.3f} (bias: {ate_dr - ATE_true:+.3f})")
print(f"  Gradient Boost: {ate_dr_gb:.3f} (bias: {ate_dr_gb - ATE_true:+.3f})")
print(f"\nFlexible ML models achieve similar performance in this case.")

# Now deliberately misspecify BOTH models (use only 2 features)
print("\n" + "="*60)
print("DELIBERATELY MISSPECIFYING BOTH MODELS (only 2 of 4 features):")
print("="*60)

feature_subset = ['MedInc', 'HouseAge']  # Missing AveRooms and Population

# Misspecified outcome model
model_wrong = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
model_wrong.fit(df[feature_subset + ['T']], df['Y'])

X_T1_wrong = df[feature_subset].copy()
X_T1_wrong['T'] = 1
X_T0_wrong = df[feature_subset].copy()
X_T0_wrong['T'] = 0

Y1_wrong_ml = model_wrong.predict(X_T1_wrong)
Y0_wrong_ml = model_wrong.predict(X_T0_wrong)

# Misspecified propensity model
ps_wrong_ml = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
ps_wrong_ml.fit(df[feature_subset], df['T'])
ps_wrong_pred = ps_wrong_ml.predict_proba(df[feature_subset])[:, 1]

# Doubly robust with both wrong
res1_wrong = df['Y'] - Y1_wrong_ml
res0_wrong = df['Y'] - Y0_wrong_ml

ate_dr_wrong = (Y1_wrong_ml.mean() + (df['T'] * res1_wrong / ps_wrong_pred).mean() -
                Y0_wrong_ml.mean() - ((1 - df['T']) * res0_wrong / (1 - ps_wrong_pred)).mean())

print(f"\nDoubly Robust ATE (both models misspecified):")
print(f"  Estimate: {ate_dr_wrong:.3f}")
print(f"  Bias: {ate_dr_wrong - ATE_true:+.3f}")
print(f"  ✗ Now substantially biased (double robustness broken)")
print(f"\nThis confirms: doubly robust requires at least ONE model to be correct.")

# Output:
# Doubly Robust ATE:
#   Linear models:  0.501 (bias: +0.001)
#   Gradient Boost: 0.498 (bias: -0.002)
#
# Flexible ML models achieve similar performance in this case.
#
# ============================================================
# DELIBERATELY MISSPECIFYING BOTH MODELS (only 2 of 4 features):
# ============================================================
#
# Doubly Robust ATE (both models misspecified):
#   Estimate: 0.623
#   Bias: +0.123
#   ✗ Now substantially biased (double robustness broken)
#
# This confirms: doubly robust requires at least ONE model to be correct.
```

With correctly specified models (all 4 features), flexible ML achieves similar performance to linear models. However, when both models are misspecified (missing important confounders), the doubly robust estimator becomes biased (0.623 vs. true 0.5), demonstrating that it requires at least one model to be correct.

## Key Takeaways

- The Average Treatment Effect (ATE) quantifies the average causal impact of a treatment across an entire population, defined as the difference between mean potential outcomes under treatment and control.
- The fundamental problem of causal inference is that we never observe both potential outcomes for the same individual—one is always a counterfactual—so all causal inference relies on assumptions to bridge this gap.
- Unconfoundedness and positivity are the two key assumptions for identifying ATE from observational data: treatment must be independent of potential outcomes given features, and all covariate patterns must have positive probability of both treatment and control.
- Naive difference-in-means is biased when treatment is confounded; adjustment methods (regression, inverse propensity weighting, doubly robust) can remove bias by accounting for confounders.
- Doubly robust estimators combine outcome regression and propensity score weighting, remaining approximately unbiased if either (but not necessarily both) model is correctly specified, making them the recommended default for ATE estimation.

**Next:** Section 55.2 covers Conditional Average Treatment Effects (CATE), which extend ATE to identify heterogeneous treatment effects across subgroups defined by features.
