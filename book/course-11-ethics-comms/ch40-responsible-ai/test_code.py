"""
Code Review Test Script for Chapter 40: Responsible AI
Tests all code blocks in sequence
"""

import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

def test_block(block_num, description, code_func):
    """Test a code block and report results."""
    print(f"\n{'='*70}")
    print(f"Testing Block {block_num}: {description}")
    print('='*70)
    try:
        code_func()
        print(f"✓ Block {block_num} PASSED")
        return True
    except Exception as e:
        print(f"✗ Block {block_num} FAILED")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return False

# Track results
results = []

# Block 1: Visualization - Bias Pipeline
def block1():
    import numpy as np
    import matplotlib.pyplot as plt

    # Create a flowchart showing where bias enters the ML pipeline
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Pipeline stages
    stages = ['Data\nCollection', 'Feature\nEngineering', 'Model\nTraining',
              'Model\nDeployment', 'Decision\nMaking']
    stage_x = np.linspace(0.1, 0.9, len(stages))
    stage_y = 0.5

    # Draw pipeline boxes
    for i, (x, stage) in enumerate(zip(stage_x, stages)):
        rect = plt.Rectangle((x - 0.06, stage_y - 0.08), 0.12, 0.16,
                              facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, stage_y, stage, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw arrows between stages
    for i in range(len(stages) - 1):
        ax.arrow(stage_x[i] + 0.06, stage_y, stage_x[i+1] - stage_x[i] - 0.12, 0,
                 head_width=0.04, head_length=0.02, fc='black', ec='black')

    # Annotate bias types
    bias_annotations = [
        (0.1, 0.75, 'Historical Bias:\nPast discrimination\nin training data', 'red'),
        (0.325, 0.75, 'Measurement Bias:\nProxy variables,\nnoisy labels', 'orange'),
        (0.55, 0.75, 'Aggregation Bias:\nOne model for\ndiverse groups', 'purple'),
        (0.775, 0.2, 'Deployment Bias:\nDifferent usage\npatterns by group', 'brown')
    ]

    for x, y, text, color in bias_annotations:
        ax.annotate(text, xy=(x, stage_y), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2),
                    fontsize=9, ha='center', color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.title('Sources of Bias in the Machine Learning Pipeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('diagrams/bias_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()

results.append(test_block(1, "Visualization - Bias Pipeline", block1))

# Block 2: Part 1 - Detecting Bias in Dataset
def block2():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_openml

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load Adult Income dataset from OpenML
    data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
    df = data.data
    df['income'] = data.target

    # Display basic information
    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nColumn types:")
    print(df.dtypes)

    # Focus on key columns for bias analysis
    df_clean = df[['age', 'education', 'sex', 'race', 'hours-per-week', 'income']].copy()
    df_clean = df_clean.dropna()

    # Convert income to binary (>50K = 1, <=50K = 0)
    df_clean['income_binary'] = (df_clean['income'] == '>50K').astype(int)

    print(f"\nCleaned dataset: {df_clean.shape[0]} rows")

    # Analyze representation bias
    print("\n--- Representation Analysis ---")
    print("\nGender distribution:")
    print(df_clean['sex'].value_counts())
    print(f"Male representation: {100 * (df_clean['sex'] == 'Male').mean():.1f}%")

    print("\nRace distribution:")
    print(df_clean['race'].value_counts())

    # Analyze outcome bias
    print("\n--- Outcome Disparities ---")
    high_income_by_gender = df_clean.groupby('sex')['income_binary'].mean()
    print("\nHigh income rate (>50K) by gender:")
    print(high_income_by_gender)
    print(f"Disparity: Males are {high_income_by_gender['Male'] / high_income_by_gender['Female']:.2f}x more likely to have high income")

    # Visualize disparities
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gender disparity
    axes[0].bar(['Female', 'Male'], high_income_by_gender[['Female', 'Male']],
                color=['coral', 'skyblue'], edgecolor='black')
    axes[0].set_ylabel('Proportion with Income >50K')
    axes[0].set_title('Income Disparity by Gender')
    axes[0].set_ylim(0, 0.4)
    for i, v in enumerate(high_income_by_gender[['Female', 'Male']]):
        axes[0].text(i, v + 0.01, f'{v:.2%}', ha='center', fontweight='bold')

    # Race disparity
    high_income_by_race = df_clean.groupby('race')['income_binary'].mean().sort_values()
    axes[1].barh(range(len(high_income_by_race)), high_income_by_race.values, color='lightgreen', edgecolor='black')
    axes[1].set_yticks(range(len(high_income_by_race)))
    axes[1].set_yticklabels(high_income_by_race.index)
    axes[1].set_xlabel('Proportion with Income >50K')
    axes[1].set_title('Income Disparity by Race')
    for i, v in enumerate(high_income_by_race.values):
        axes[1].text(v + 0.005, i, f'{v:.2%}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('diagrams/adult_income_bias.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Store for next block
    global df_clean_global
    df_clean_global = df_clean

results.append(test_block(2, "Part 1 - Detecting Bias", block2))

# Block 3: Part 2 - Training Model and Computing Fairness Metrics
def block3():
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    df_clean = df_clean_global

    # Prepare features for modeling
    le_education = LabelEncoder()
    le_race = LabelEncoder()
    le_sex = LabelEncoder()

    X = df_clean.copy()
    X['education_encoded'] = le_education.fit_transform(X['education'])
    X['race_encoded'] = le_race.fit_transform(X['race'])
    X['sex_encoded'] = le_sex.fit_transform(X['sex'])

    # Select features
    feature_cols = ['age', 'hours-per-week', 'education_encoded', 'race_encoded']
    X_model = X[feature_cols]
    y = X['income_binary']
    sensitive_feature = X['sex']

    # Split data
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X_model, y, sensitive_feature, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train baseline logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nBaseline Model Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

    # Compute fairness metrics
    female_mask = (sensitive_test == 'Female')
    male_mask = (sensitive_test == 'Male')

    # Demographic parity
    positive_rate_female = y_pred[female_mask].mean()
    positive_rate_male = y_pred[male_mask].mean()

    print("\n--- Fairness Analysis ---")
    print(f"\nPositive prediction rate (ŷ=1):")
    print(f"  Female: {positive_rate_female:.3f}")
    print(f"  Male: {positive_rate_male:.3f}")
    print(f"  Demographic Parity Difference: {abs(positive_rate_male - positive_rate_female):.3f}")

    # Disparate impact ratio
    disparate_impact = positive_rate_female / positive_rate_male
    print(f"  Disparate Impact Ratio: {disparate_impact:.3f}")
    if disparate_impact < 0.8:
        print("  ⚠️  WARNING: Disparate impact ratio < 0.8 indicates significant bias")

    # Equalized odds
    female_qualified = female_mask & (y_test == 1)
    male_qualified = male_mask & (y_test == 1)
    tpr_female = y_pred[female_qualified].mean() if female_qualified.sum() > 0 else 0
    tpr_male = y_pred[male_qualified].mean() if male_qualified.sum() > 0 else 0

    female_unqualified = female_mask & (y_test == 0)
    male_unqualified = male_mask & (y_test == 0)
    fpr_female = y_pred[female_unqualified].mean() if female_unqualified.sum() > 0 else 0
    fpr_male = y_pred[male_unqualified].mean() if male_unqualified.sum() > 0 else 0

    print(f"\nTrue Positive Rate (TPR):")
    print(f"  Female: {tpr_female:.3f}")
    print(f"  Male: {tpr_male:.3f}")
    print(f"  TPR Difference: {abs(tpr_male - tpr_female):.3f}")

    print(f"\nFalse Positive Rate (FPR):")
    print(f"  Female: {fpr_female:.3f}")
    print(f"  Male: {fpr_male:.3f}")
    print(f"  FPR Difference: {abs(fpr_male - fpr_female):.3f}")

    avg_odds_diff = (abs(tpr_male - tpr_female) + abs(fpr_male - fpr_female)) / 2
    print(f"\nAverage Odds Difference: {avg_odds_diff:.3f}")

    # Store for next block
    global fairness_data
    fairness_data = {
        'y_proba': y_proba,
        'y_test': y_test,
        'sensitive_test': sensitive_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'positive_rate_female': positive_rate_female,
        'positive_rate_male': positive_rate_male,
        'tpr_female': tpr_female,
        'tpr_male': tpr_male,
        'female_mask': female_mask,
        'male_mask': male_mask,
        'female_qualified': female_qualified,
        'male_qualified': male_qualified,
        'female_unqualified': female_unqualified,
        'male_unqualified': male_unqualified
    }

results.append(test_block(3, "Part 2 - Training Model & Fairness", block3))

# Block 4: Part 3 - Debiasing with Threshold Optimization
def block4():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import accuracy_score

    y_proba = fairness_data['y_proba']
    y_test = fairness_data['y_test']
    sensitive_test = fairness_data['sensitive_test']
    y_pred = fairness_data['y_pred']
    accuracy = fairness_data['accuracy']
    positive_rate_female = fairness_data['positive_rate_female']
    positive_rate_male = fairness_data['positive_rate_male']
    tpr_female = fairness_data['tpr_female']
    tpr_male = fairness_data['tpr_male']
    female_mask = fairness_data['female_mask']
    male_mask = fairness_data['male_mask']
    female_qualified = fairness_data['female_qualified']
    male_qualified = fairness_data['male_qualified']
    female_unqualified = fairness_data['female_unqualified']
    male_unqualified = fairness_data['male_unqualified']

    def find_threshold_for_tpr(probabilities, true_labels, target_tpr=0.65):
        """Find probability threshold that achieves target true positive rate."""
        thresholds = np.linspace(0, 1, 100)
        best_threshold = 0.5
        best_diff = float('inf')

        for thresh in thresholds:
            predictions = (probabilities >= thresh).astype(int)
            if true_labels.sum() > 0:
                tpr = predictions[true_labels == 1].mean()
                diff = abs(tpr - target_tpr)
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = thresh

        return best_threshold

    # Separate probabilities by group
    proba_female = y_proba[female_mask]
    proba_male = y_proba[male_mask]
    y_test_female = y_test[female_mask]
    y_test_male = y_test[male_mask]

    # Find thresholds
    target_tpr = 0.65
    threshold_female = find_threshold_for_tpr(proba_female, y_test_female, target_tpr)
    threshold_male = find_threshold_for_tpr(proba_male, y_test_male, target_tpr)

    print(f"\n--- Threshold Optimization ---")
    print(f"Target TPR: {target_tpr:.3f}")
    print(f"Optimal threshold for females: {threshold_female:.3f}")
    print(f"Optimal threshold for males: {threshold_male:.3f}")

    # Apply group-specific thresholds
    y_pred_fair = np.zeros_like(y_pred)
    y_pred_fair[female_mask] = (proba_female >= threshold_female).astype(int)
    y_pred_fair[male_mask] = (proba_male >= threshold_male).astype(int)

    # Evaluate fairness after debiasing
    accuracy_fair = accuracy_score(y_test, y_pred_fair)
    positive_rate_female_fair = y_pred_fair[female_mask].mean()
    positive_rate_male_fair = y_pred_fair[male_mask].mean()

    tpr_female_fair = y_pred_fair[female_qualified].mean() if female_qualified.sum() > 0 else 0
    tpr_male_fair = y_pred_fair[male_qualified].mean() if male_qualified.sum() > 0 else 0
    fpr_female_fair = y_pred_fair[female_unqualified].mean() if female_unqualified.sum() > 0 else 0
    fpr_male_fair = y_pred_fair[male_unqualified].mean() if male_unqualified.sum() > 0 else 0

    print(f"\n--- Results After Debiasing ---")
    print(f"Accuracy: {accuracy_fair:.3f} (baseline: {accuracy:.3f}, change: {accuracy_fair - accuracy:+.3f})")
    print(f"\nDemographic Parity Difference: {abs(positive_rate_male_fair - positive_rate_female_fair):.3f} (baseline: {abs(positive_rate_male - positive_rate_female):.3f})")
    print(f"\nTrue Positive Rate:")
    print(f"  Female: {tpr_female_fair:.3f} (baseline: {tpr_female:.3f})")
    print(f"  Male: {tpr_male_fair:.3f} (baseline: {tpr_male:.3f})")
    print(f"  TPR Difference: {abs(tpr_male_fair - tpr_female_fair):.3f} (baseline: {abs(tpr_male - tpr_female):.3f})")

    # Visualize the trade-off
    metrics_comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Demo. Parity Diff', 'TPR Difference'],
        'Baseline': [accuracy, abs(positive_rate_male - positive_rate_female), abs(tpr_male - tpr_female)],
        'After Debiasing': [accuracy_fair, abs(positive_rate_male_fair - positive_rate_female_fair), abs(tpr_male_fair - tpr_female_fair)]
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_comparison))
    width = 0.35

    bars1 = ax.bar(x - width/2, metrics_comparison['Baseline'], width, label='Baseline', color='lightcoral', edgecolor='black')
    bars2 = ax.bar(x + width/2, metrics_comparison['After Debiasing'], width, label='After Debiasing', color='lightgreen', edgecolor='black')

    ax.set_ylabel('Metric Value')
    ax.set_title('Accuracy-Fairness Trade-off: Baseline vs. Debiased Model')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_comparison['Metric'])
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('diagrams/fairness_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

results.append(test_block(4, "Part 3 - Debiasing", block4))

# Block 5: Part 4 - Simulating Differential Privacy
def block5():
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n_employees = 500
    true_salaries = np.random.normal(loc=75000, scale=15000, size=n_employees)
    true_mean_salary = true_salaries.mean()

    print(f"Dataset: {n_employees} employees")
    print(f"True mean salary: ${true_mean_salary:,.2f}")

    def laplace_mechanism(true_value, sensitivity, epsilon):
        scale = sensitivity / epsilon
        noise = np.random.laplace(loc=0, scale=scale)
        return true_value + noise

    salary_range = 200000 - 20000
    sensitivity = salary_range / n_employees

    print(f"\nSensitivity (for mean query): ${sensitivity:,.2f}")

    epsilons = [0.1, 1.0, 10.0]
    n_queries = 100

    results_dp = {}
    for epsilon in epsilons:
        noisy_means = [laplace_mechanism(true_mean_salary, sensitivity, epsilon)
                       for _ in range(n_queries)]
        results_dp[epsilon] = noisy_means

        print(f"\nε = {epsilon}:")
        print(f"  Mean of noisy queries: ${np.mean(noisy_means):,.2f}")
        print(f"  Std of noisy queries: ${np.std(noisy_means):,.2f}")
        print(f"  Typical error: ±${np.std(noisy_means):,.2f}")
        print(f"  Privacy: {'Strong' if epsilon < 1 else 'Moderate' if epsilon < 5 else 'Weak'}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, epsilon in enumerate(epsilons):
        ax = axes[i]
        ax.hist(results_dp[epsilon], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(true_mean_salary, color='red', linestyle='--', linewidth=2,
                   label=f'True mean: ${true_mean_salary:,.0f}')
        ax.set_xlabel('Noisy Mean Salary ($)')
        ax.set_ylabel('Frequency (out of 100 queries)')
        ax.set_title(f'ε = {epsilon}\n({"Strong" if epsilon < 1 else "Moderate" if epsilon < 5 else "Weak"} Privacy)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('diagrams/differential_privacy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Demonstrate privacy guarantee
    print("\n--- Privacy Guarantee Demonstration ---")
    modified_salaries = np.append(true_salaries, 500000)
    modified_mean = modified_salaries.mean()

    epsilon_demo = 1.0
    noisy_original = laplace_mechanism(true_mean_salary, sensitivity, epsilon_demo)
    noisy_modified = laplace_mechanism(modified_mean, sensitivity, epsilon_demo)

    print(f"\nOriginal dataset mean: ${true_mean_salary:,.2f}")
    print(f"Modified dataset mean (with CEO): ${modified_mean:,.2f}")
    print(f"Actual difference: ${abs(modified_mean - true_mean_salary):,.2f}")
    print(f"\nNoisy query on original: ${noisy_original:,.2f}")
    print(f"Noisy query on modified: ${noisy_modified:,.2f}")
    print(f"Noisy difference: ${abs(noisy_modified - noisy_original):,.2f}")
    print(f"\nWith ε={epsilon_demo}, an adversary cannot reliably determine if the CEO's record is in the dataset.")
    print(f"The noise (±${sensitivity/epsilon_demo:,.2f} typical) masks individual contributions.")

results.append(test_block(5, "Part 4 - Differential Privacy", block5))

# Print summary
print("\n" + "="*70)
print("CODE REVIEW SUMMARY")
print("="*70)
print(f"Total blocks tested: {len(results)}")
print(f"Passed: {sum(results)}")
print(f"Failed: {len(results) - sum(results)}")
print("="*70)
