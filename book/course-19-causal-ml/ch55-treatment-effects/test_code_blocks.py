"""
Code Review Test Script for Chapter 55: Average Treatment Effect (ATE)
This script executes all code blocks from content.md to verify they work correctly.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Track test results
test_results = {
    'blocks_tested': 0,
    'blocks_passed': 0,
    'failures': [],
    'style_issues': [],
    'dependencies': set()
}

def test_block(block_name, code_func):
    """Test a single code block"""
    global test_results
    test_results['blocks_tested'] += 1
    try:
        print(f"\n{'='*60}")
        print(f"Testing: {block_name}")
        print('='*60)
        code_func()
        test_results['blocks_passed'] += 1
        print(f"✓ {block_name} PASSED")
    except Exception as e:
        test_results['failures'].append((block_name, str(e)))
        print(f"✗ {block_name} FAILED: {str(e)}")

# Create diagrams directory
import os
os.makedirs('diagrams', exist_ok=True)

# ============================================================================
# VISUALIZATION BLOCK
# ============================================================================
def block_1_visualization():
    """Block 1: Potential Outcomes Framework Visualization"""
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
    plt.close()

test_block("Block 1: Visualization", block_1_visualization)

# ============================================================================
# EXAMPLES - PART 1: Simulating Data
# ============================================================================
def block_2_simulate_data():
    """Block 2: Simulating treatment effect data with confounding"""
    global df, T, Y, X, ATE_true

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
    Y_0 = (2.0 * X['MedInc'] +
           1.5 * X['HouseAge'] +
           0.8 * X['AveRooms'] -
           0.5 * X['Population'] +
           np.random.normal(0, 0.5, n))

    Y_1 = Y_0 + ATE_true  # Treatment adds constant effect

    # Observed outcome (fundamental problem: only see one potential outcome)
    Y = T * Y_1 + (1 - T) * Y_0

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
    print(f"Naive difference in means: {Y[T==1].mean() - Y[T==0].mean():.3f}")
    print(f"(Biased due to confounding!)")

test_block("Block 2: Simulating Data", block_2_simulate_data)

# ============================================================================
# PART 2: Checking Overlap
# ============================================================================
def block_3_overlap_check():
    """Block 3: Visualize propensity score overlap"""
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
    plt.close()

    print("\nPositivity Check:")
    print(f"Propensity scores range: [{df['propensity_score'].min():.3f}, {df['propensity_score'].max():.3f}]")
    print(f"Proportion with extreme scores (<0.1 or >0.9): {((df['propensity_score'] < 0.1) | (df['propensity_score'] > 0.9)).mean()*100:.1f}%")
    print("\nCovariate Balance (Standardized Mean Differences):")
    for feature, smd_val in zip(features_to_check, smd):
        status = "⚠ IMBALANCED" if abs(smd_val) > 0.1 else "✓ Balanced"
        print(f"  {feature:15s}: {smd_val:6.3f}  {status}")

test_block("Block 3: Overlap Check", block_3_overlap_check)

# ============================================================================
# PART 3: Estimating ATE with Multiple Methods
# ============================================================================
def block_4_ate_estimation():
    """Block 4: Estimating ATE with multiple methods"""
    global ate_dr, ps_pred, Y_1_pred, Y_0_pred, feature_cols

    # Method 1: Naive difference in means (BIASED)
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
    plt.close()

test_block("Block 4: ATE Estimation", block_4_ate_estimation)

# ============================================================================
# PART 4: Understanding Why Doubly Robust Works
# ============================================================================
def block_5_doubly_robust():
    """Block 5: Demonstrate robustness of doubly robust estimator"""
    n = len(df)

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

test_block("Block 5: Doubly Robust Demo", block_5_doubly_robust)

# ============================================================================
# SOLUTION 1: Diabetes Dataset
# ============================================================================
def block_6_solution_1():
    """Block 6: Solution 1 - Diabetes dataset ATE estimation"""
    # Load diabetes dataset
    diabetes = load_diabetes()
    X_diab = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y_diab = diabetes.target

    # Create binary treatment based on BMI
    T_diab = (X_diab['bmi'] > X_diab['bmi'].median()).astype(int)

    # Prepare data
    df_diabetes = X_diab.copy()
    df_diabetes['T'] = T_diab
    df_diabetes['Y'] = y_diab

    print("Treatment Distribution:")
    print(df_diabetes['T'].value_counts())

    # Method 1: Naive difference-in-means
    ate_naive = df_diabetes[df_diabetes['T']==1]['Y'].mean() - df_diabetes[df_diabetes['T']==0]['Y'].mean()

    # Method 2: Regression adjustment
    # Control for age, sex, bp
    covariates = ['age', 'sex', 'bp']
    X_reg = df_diabetes[covariates + ['T']].values
    model_reg = LinearRegression()
    model_reg.fit(X_reg, df_diabetes['Y'])

    # Predict under both treatment conditions
    X_T1 = df_diabetes[covariates].copy()
    X_T1['T'] = 1
    X_T0 = df_diabetes[covariates].copy()
    X_T0['T'] = 0

    Y1_pred_sol = model_reg.predict(X_T1)
    Y0_pred_sol = model_reg.predict(X_T0)
    ate_reg = (Y1_pred_sol - Y0_pred_sol).mean()

    # Method 3: Inverse Propensity Weighting
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(df_diabetes[covariates], df_diabetes['T'])
    ps_sol = ps_model.predict_proba(df_diabetes[covariates])[:, 1]

    # IPW weights
    weights = df_diabetes['T'] / ps_sol + (1 - df_diabetes['T']) / (1 - ps_sol)

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

test_block("Block 6: Solution 1", block_6_solution_1)

# ============================================================================
# SOLUTION 2: Overlap and Balance Plots
# ============================================================================
def block_7_solution_2():
    """Block 7: Solution 2 - Propensity score overlap and covariate balance"""
    # Reload diabetes data
    diabetes = load_diabetes()
    X_diab = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y_diab = diabetes.target
    T_diab = (X_diab['bmi'] > X_diab['bmi'].median()).astype(int)
    df_diabetes = X_diab.copy()
    df_diabetes['T'] = T_diab
    df_diabetes['Y'] = y_diab

    # Fit propensity model
    covariates = ['age', 'sex', 'bp']
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(df_diabetes[covariates], df_diabetes['T'])
    ps = ps_model.predict_proba(df_diabetes[covariates])[:, 1]

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
    weights = df_diabetes['T'] / ps + (1 - df_diabetes['T']) / (1 - ps)
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
    ax2.set_title('Covariate Balance: Before and After IPW')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.close()

    print(f"\nPositivity Check:")
    print(f"  Propensity score range: [{ps.min():.3f}, {ps.max():.3f}]")
    print(f"  Extreme scores (<0.1 or >0.9): {((ps < 0.1) | (ps > 0.9)).sum()} observations")
    print(f"\nPositivity holds well. Balance is improved after IPW (points move toward 0).")

test_block("Block 7: Solution 2", block_7_solution_2)

# ============================================================================
# SOLUTION 3: Simulation Study
# ============================================================================
def block_8_solution_3():
    """Block 8: Solution 3 - Simulation study"""
    np.random.seed(42)

    def simulate_once(n=1000, tau=1.5):
        # Generate confounders
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)

        # Treatment assignment
        propensity_logit = X1 + X2
        ps_true = 1 / (1 + np.exp(-propensity_logit))
        T_sim = np.random.binomial(1, ps_true)

        # Outcome
        Y_sim = 2*X1 + 3*X2 + T_sim*tau + np.random.normal(0, 1, n)

        # Estimate ATE with four methods

        # 1. Naive
        ate_naive = Y_sim[T_sim==1].mean() - Y_sim[T_sim==0].mean()

        # 2. Regression
        X_reg = np.column_stack([X1, X2, T_sim])
        model_reg = LinearRegression()
        model_reg.fit(X_reg, Y_sim)
        X_T1 = np.column_stack([X1, X2, np.ones(n)])
        X_T0 = np.column_stack([X1, X2, np.zeros(n)])
        ate_reg = (model_reg.predict(X_T1) - model_reg.predict(X_T0)).mean()

        # 3. IPW
        X_ps = np.column_stack([X1, X2])
        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(X_ps, T_sim)
        ps_pred_sim = ps_model.predict_proba(X_ps)[:, 1]
        weights = T_sim / ps_pred_sim + (1 - T_sim) / (1 - ps_pred_sim)
        ate_ipw = (Y_sim * T_sim * weights).sum() / (T_sim * weights).sum() - (Y_sim * (1-T_sim) * weights).sum() / ((1-T_sim) * weights).sum()

        # 4. Doubly Robust
        Y1_pred_sim = model_reg.predict(X_T1)
        Y0_pred_sim = model_reg.predict(X_T0)
        ate_dr = (Y1_pred_sim.mean() + (T_sim * (Y_sim - Y1_pred_sim) / ps_pred_sim).mean() -
                  Y0_pred_sim.mean() - ((1-T_sim) * (Y_sim - Y0_pred_sim) / (1 - ps_pred_sim)).mean())

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
    plt.close()

test_block("Block 8: Solution 3", block_8_solution_3)

# ============================================================================
# SOLUTION 4: Bootstrap Confidence Intervals
# ============================================================================
def block_9_solution_4():
    """Block 9: Solution 4 - Bootstrap confidence intervals"""
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
    plt.close()

test_block("Block 9: Solution 4", block_9_solution_4)

# ============================================================================
# SOLUTION 5: Machine Learning Models
# ============================================================================
def block_10_solution_5():
    """Block 10: Solution 5 - Doubly robust with ML models"""
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

test_block("Block 10: Solution 5", block_10_solution_5)

# ============================================================================
# SUMMARY AND DEPENDENCIES
# ============================================================================
print("\n" + "="*70)
print("CODE REVIEW SUMMARY")
print("="*70)

test_dependencies = [
    'numpy', 'pandas', 'matplotlib', 'seaborn',
    'sklearn.datasets.fetch_california_housing',
    'sklearn.datasets.load_diabetes',
    'sklearn.linear_model.LinearRegression',
    'sklearn.linear_model.LogisticRegression',
    'sklearn.ensemble.GradientBoostingClassifier',
    'sklearn.ensemble.GradientBoostingRegressor',
    'sklearn.utils.resample'
]

print(f"\nBlocks Tested: {test_results['blocks_passed']}/{test_results['blocks_tested']}")

if test_results['failures']:
    print(f"\n❌ FAILURES ({len(test_results['failures'])}):")
    for i, (block, error) in enumerate(test_results['failures'], 1):
        print(f"\n{i}. {block}:")
        print(f"   Error: {error[:200]}...")
else:
    print("\n✓ ALL TESTS PASSED")

print(f"\nDependencies Used:")
for dep in test_dependencies:
    print(f"  - {dep}")

print("\n" + "="*70)

# Determine rating
if test_results['blocks_passed'] == test_results['blocks_tested']:
    rating = "ALL_PASS"
elif test_results['blocks_passed'] >= test_results['blocks_tested'] * 0.8:
    rating = "MINOR_FIXES"
else:
    rating = "BROKEN"

print(f"\n🏆 RATING: {rating}")
print("="*70)
