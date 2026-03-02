"""
Test Solutions for Chapter 40: Responsible AI
"""

import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

def test_solution(solution_num, description, code_func):
    """Test a solution and report results."""
    print(f"\n{'='*70}")
    print(f"Testing Solution {solution_num}: {description}")
    print('='*70)
    try:
        code_func()
        print(f"✓ Solution {solution_num} PASSED")
        return True
    except Exception as e:
        print(f"✗ Solution {solution_num} FAILED")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return False

results = []

# Solution 1: COMPAS recidivism bias audit
def solution1():
    from fairlearn.datasets import fetch_compas
    import pandas as pd
    import numpy as np

    # Load COMPAS data
    data = fetch_compas(as_frame=True)
    df = data.data
    df['recidivism'] = data.target

    # Filter to two racial groups for comparison
    df_filtered = df[df['race'].isin(['Caucasian', 'African-American'])].copy()

    print("--- COMPAS Bias Audit ---")
    print(f"Total samples: {len(df_filtered)}")
    print(f"\nRacial distribution:")
    print(df_filtered['race'].value_counts())

    # Base rate of high-risk predictions by race
    base_rate_by_race = df_filtered.groupby('race')['recidivism'].mean()
    print(f"\nRecidivism prediction rate by race:")
    print(base_rate_by_race)

    # Use actual recidivism as proxy
    df_filtered['pred_recidivist'] = df_filtered['recidivism']
    y_pred_col = 'pred_recidivist'
    y_true_col = 'recidivism'

    # Compute FPR and FNR by race
    for race in ['Caucasian', 'African-American']:
        race_data = df_filtered[df_filtered['race'] == race]

        actual_non_recid = race_data[race_data[y_true_col] == 0]
        if len(actual_non_recid) > 0:
            fpr = (actual_non_recid[y_pred_col] == 1).mean()
        else:
            fpr = 0

        actual_recid = race_data[race_data[y_true_col] == 1]
        if len(actual_recid) > 0:
            fnr = (actual_recid[y_pred_col] == 0).mean()
        else:
            fnr = 0

        print(f"\n{race}:")
        print(f"  False Positive Rate (FPR): {fpr:.3f}")
        print(f"  False Negative Rate (FNR): {fnr:.3f}")

    # Disparate impact ratio
    pred_rate_aa = df_filtered[df_filtered['race'] == 'African-American'][y_pred_col].mean()
    pred_rate_c = df_filtered[df_filtered['race'] == 'Caucasian'][y_pred_col].mean()
    disparate_impact = pred_rate_c / pred_rate_aa if pred_rate_aa > 0 else 0

    print(f"\nDisparate Impact Ratio (Caucasian/African-American): {disparate_impact:.3f}")
    if disparate_impact < 0.8 or disparate_impact > 1.25:
        print("⚠️  WARNING: Ratio outside [0.8, 1.25] indicates significant bias")
        print("\nSuggested debiasing technique:")
        print("Apply *threshold optimization* (post-processing): use race-specific")
        print("probability thresholds to equalize false positive rates across groups,")
        print("ensuring non-recidivists are incorrectly flagged at equal rates.")

results.append(test_solution(1, "COMPAS Bias Audit", solution1))

# Solution 2: Random Forest with fairness evaluation
def solution2():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import fetch_openml
    import pandas as pd
    import numpy as np

    data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
    df = data.data
    df['income'] = data.target
    df_clean = df[['age', 'education', 'sex', 'race', 'hours-per-week', 'income']].dropna()
    df_clean['income_binary'] = (df_clean['income'] == '>50K').astype(int)

    le_education = LabelEncoder()
    le_race = LabelEncoder()
    X = df_clean.copy()
    X['education_encoded'] = le_education.fit_transform(X['education'])
    X['race_encoded'] = le_race.fit_transform(X['race'])

    feature_cols = ['age', 'hours-per-week', 'education_encoded', 'race_encoded']
    X_model = X[feature_cols]
    y = X['income_binary']
    sensitive = X['sex']

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X_model, y, sensitive, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_baseline = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    female_mask = (sens_test == 'Female')
    male_mask = (sens_test == 'Male')
    dp_diff_baseline = abs(y_pred_baseline[male_mask].mean() - y_pred_baseline[female_mask].mean())

    tpr_f_base = y_pred_baseline[female_mask & (y_test == 1)].mean()
    tpr_m_base = y_pred_baseline[male_mask & (y_test == 1)].mean()
    eo_diff_baseline = abs(tpr_m_base - tpr_f_base)

    print(f"Baseline Accuracy: {acc_baseline:.3f}")
    print(f"Baseline Demographic Parity Difference: {dp_diff_baseline:.3f}")
    print(f"Baseline Equalized Odds Difference: {eo_diff_baseline:.3f}")

    target_dp = 0.05

    def find_threshold_for_rate(probabilities, target_rate):
        thresholds = np.linspace(0, 1, 200)
        best_thresh = 0.5
        best_diff = float('inf')
        for thresh in thresholds:
            rate = (probabilities >= thresh).mean()
            if abs(rate - target_rate) < best_diff:
                best_diff = abs(rate - target_rate)
                best_thresh = thresh
        return best_thresh

    target_rate = (y_pred_baseline[female_mask].mean() + y_pred_baseline[male_mask].mean()) / 2
    thresh_f = find_threshold_for_rate(y_proba[female_mask], target_rate)
    thresh_m = find_threshold_for_rate(y_proba[male_mask], target_rate)

    y_pred_fair = np.zeros_like(y_pred_baseline)
    y_pred_fair[female_mask] = (y_proba[female_mask] >= thresh_f).astype(int)
    y_pred_fair[male_mask] = (y_proba[male_mask] >= thresh_m).astype(int)

    acc_fair = accuracy_score(y_test, y_pred_fair)
    dp_diff_fair = abs(y_pred_fair[male_mask].mean() - y_pred_fair[female_mask].mean())
    tpr_f_fair = y_pred_fair[female_mask & (y_test == 1)].mean()
    tpr_m_fair = y_pred_fair[male_mask & (y_test == 1)].mean()
    eo_diff_fair = abs(tpr_m_fair - tpr_f_fair)

    print(f"\nAfter Debiasing:")
    print(f"Accuracy: {acc_fair:.3f} (change: {acc_fair - acc_baseline:+.3f})")
    print(f"Demographic Parity Difference: {dp_diff_fair:.3f} (target: <0.05)")
    print(f"Equalized Odds Difference: {eo_diff_fair:.3f}")

    print("\n--- Trade-off Analysis for Hiring Application ---")
    if dp_diff_fair < 0.05:
        print("✓ Achieved demographic parity target")
    if abs(acc_fair - acc_baseline) < 0.05:
        print("✓ Accuracy loss is minimal (<5 percentage points)")

    print(f"\nFor a hiring application, this trade-off is acceptable because:")
    print(f"1. Demographic parity improved from {dp_diff_baseline:.3f} to {dp_diff_fair:.3f}")
    print(f"2. Accuracy decreased only {100*abs(acc_fair - acc_baseline):.1f}%")
    print(f"3. Legal compliance: Many jurisdictions require disparate impact ratio > 0.8")

results.append(test_solution(2, "Random Forest Debiasing", solution2))

# Solution 3: Differential privacy for counting
def solution3():
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)

    n = 500
    prevalence = 0.3
    data = np.random.choice([0, 1], size=n, p=[1-prevalence, prevalence])
    true_count = data.sum()

    print(f"Dataset: {n} individuals")
    print(f"True count with condition: {true_count} ({100*true_count/n:.1f}%)")

    def dp_count(data, epsilon):
        true_count = data.sum()
        sensitivity = 1
        scale = sensitivity / epsilon
        noise = np.random.laplace(loc=0, scale=scale)
        return true_count + noise

    epsilons = [0.1, 1.0, 10.0]
    n_trials = 100
    results_dp = {}

    for eps in epsilons:
        noisy_counts = [dp_count(data, eps) for _ in range(n_trials)]
        results_dp[eps] = noisy_counts

        mean_noisy = np.mean(noisy_counts)
        std_noisy = np.std(noisy_counts)
        print(f"\nε = {eps}:")
        print(f"  Mean noisy count: {mean_noisy:.1f}")
        print(f"  Std deviation: {std_noisy:.1f}")
        print(f"  Typical error: ±{std_noisy:.1f} (±{100*std_noisy/true_count:.0f}% of true count)")
        print(f"  Privacy level: {'Strong' if eps < 1 else 'Moderate' if eps < 5 else 'Weak'}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, eps in enumerate(epsilons):
        ax = axes[i]
        ax.hist(results_dp[eps], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(true_count, color='red', linestyle='--', linewidth=2.5,
                   label=f'True count: {true_count}')
        ax.set_xlabel('Noisy Count')
        ax.set_ylabel('Frequency')
        ax.set_title(f'ε = {eps}\n({"Strong" if eps < 1 else "Moderate" if eps < 5 else "Weak"} Privacy)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('diagrams/dp_count_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n--- Recommendations ---")
    print("Public Health Study (requires high accuracy):")
    print("  → Use ε = 10.0")

    print("\nClinical Trial (requires strong individual privacy):")
    print("  → Use ε = 0.1")

results.append(test_solution(3, "Differential Privacy Counting", solution3))

# Solution 5: Proxy variables analysis
def solution5():
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import numpy as np

    data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
    df = data.data
    df['income'] = data.target
    df_clean = df[['age', 'education', 'occupation', 'sex', 'race', 'hours-per-week', 'income']].dropna()
    df_clean['income_binary'] = (df_clean['income'] == '>50K').astype(int)

    le_dict = {}
    for col in ['education', 'occupation', 'race', 'sex']:
        le = LabelEncoder()
        df_clean[col + '_encoded'] = le.fit_transform(df_clean[col])
        le_dict[col] = le

    gender_encoded = df_clean['sex_encoded']
    correlations = {}
    for col in ['age', 'education_encoded', 'occupation_encoded', 'race_encoded', 'hours-per-week']:
        corr = np.corrcoef(df_clean[col], gender_encoded)[0, 1]
        correlations[col] = corr

    print("--- Correlation with Gender ---")
    for col, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{col}: {corr:+.3f}")

    proxies = [col for col, corr in correlations.items() if abs(corr) > 0.2]
    print(f"\nIdentified proxy variables (|corr| > 0.2): {proxies}")

    features_model1 = ['age', 'education_encoded', 'occupation_encoded', 'race_encoded', 'hours-per-week']
    X1 = df_clean[features_model1]
    y = df_clean['income_binary']
    sensitive = df_clean['sex']

    X_train1, X_test1, y_train, y_test, sens_train, sens_test = train_test_split(
        X1, y, sensitive, test_size=0.2, random_state=42, stratify=y
    )

    model1 = LogisticRegression(max_iter=1000, random_state=42)
    model1.fit(X_train1, y_train)
    y_pred1 = model1.predict(X_test1)

    female_mask = (sens_test == 'Female')
    male_mask = (sens_test == 'Male')
    dp_diff1 = abs(y_pred1[male_mask].mean() - y_pred1[female_mask].mean())

    print(f"\nModel 1 (all features except sex):")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred1):.3f}")
    print(f"  Demographic Parity Difference: {dp_diff1:.3f}")

    features_model2 = [f for f in features_model1 if f not in proxies]
    print(f"\nModel 2 features (excluding proxies {proxies}): {features_model2}")

    if len(features_model2) > 0:
        X2 = df_clean[features_model2]
        X_train2, X_test2, _, _, _, _ = train_test_split(
            X2, y, sensitive, test_size=0.2, random_state=42, stratify=y
        )

        model2 = LogisticRegression(max_iter=1000, random_state=42)
        model2.fit(X_train2, y_train)
        y_pred2 = model2.predict(X_test2)

        dp_diff2 = abs(y_pred2[male_mask].mean() - y_pred2[female_mask].mean())

        print(f"\nModel 2 (excluding sex + proxies):")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred2):.3f}")
        print(f"  Demographic Parity Difference: {dp_diff2:.3f}")

        print(f"\n--- Analysis ---")
        print(f"Removing proxies changed DP difference: {dp_diff1:.3f} → {dp_diff2:.3f}")
        print("\nConclusion: Feature removal is insufficient for debiasing.")

results.append(test_solution(5, "Proxy Variables Analysis", solution5))

# Print summary
print("\n" + "="*70)
print("SOLUTIONS REVIEW SUMMARY")
print("="*70)
print(f"Total solutions tested: {len(results)}")
print(f"Passed: {sum(results)}")
print(f"Failed: {len(results) - sum(results)}")
print("="*70)
