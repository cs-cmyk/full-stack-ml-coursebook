"""
Generate all diagrams for Chapter 40: Responsible AI
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set style
plt.style.use('default')
np.random.seed(42)

# Color palette
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

print("Generating diagrams for Chapter 40: Responsible AI...")

# ============================================================================
# DIAGRAM 1: Bias Pipeline
# ============================================================================
print("\n1. Generating bias_pipeline.png...")

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
                          facecolor=COLORS['blue'], edgecolor='black', linewidth=2, alpha=0.3)
    ax.add_patch(rect)
    ax.text(x, stage_y, stage, ha='center', va='center', fontsize=12, fontweight='bold')

# Draw arrows between stages
for i in range(len(stages) - 1):
    ax.arrow(stage_x[i] + 0.06, stage_y, stage_x[i+1] - stage_x[i] - 0.12, 0,
             head_width=0.04, head_length=0.02, fc='black', ec='black')

# Annotate bias types
bias_annotations = [
    (0.1, 0.75, 'Historical Bias:\nPast discrimination\nin training data', COLORS['red']),
    (0.325, 0.75, 'Measurement Bias:\nProxy variables,\nnoisy labels', COLORS['orange']),
    (0.55, 0.75, 'Aggregation Bias:\nOne model for\ndiverse groups', COLORS['purple']),
    (0.775, 0.2, 'Deployment Bias:\nDifferent usage\npatterns by group', COLORS['gray'])
]

for x, y, text, color in bias_annotations:
    ax.annotate(text, xy=(x, stage_y), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=2),
                fontsize=10, ha='center', color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color, linewidth=2))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.title('Sources of Bias in the Machine Learning Pipeline', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-11-ethics-comms/ch40-responsible-ai/diagrams/bias_pipeline.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ bias_pipeline.png created")

# ============================================================================
# DIAGRAM 2: Adult Income Bias Analysis
# ============================================================================
print("\n2. Generating adult_income_bias.png...")

# Load Adult Income dataset
data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
df = data.data
df['income'] = data.target

# Clean data
df_clean = df[['age', 'education', 'sex', 'race', 'hours-per-week', 'income']].copy()
df_clean = df_clean.dropna()
df_clean['income_binary'] = (df_clean['income'] == '>50K').astype(int)

# Analyze disparities
high_income_by_gender = df_clean.groupby('sex')['income_binary'].mean()
high_income_by_race = df_clean.groupby('race')['income_binary'].mean().sort_values()

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gender disparity
axes[0].bar(['Female', 'Male'], high_income_by_gender[['Female', 'Male']],
            color=[COLORS['orange'], COLORS['blue']], edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('Proportion with Income >50K', fontsize=12, fontweight='bold')
axes[0].set_title('Income Disparity by Gender', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 0.4)
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(high_income_by_gender[['Female', 'Male']]):
    axes[0].text(i, v + 0.01, f'{v:.2%}', ha='center', fontweight='bold', fontsize=11)

# Race disparity
axes[1].barh(range(len(high_income_by_race)), high_income_by_race.values,
             color=COLORS['green'], edgecolor='black', linewidth=1.5, alpha=0.7)
axes[1].set_yticks(range(len(high_income_by_race)))
axes[1].set_yticklabels(high_income_by_race.index, fontsize=10)
axes[1].set_xlabel('Proportion with Income >50K', fontsize=12, fontweight='bold')
axes[1].set_title('Income Disparity by Race', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)
for i, v in enumerate(high_income_by_race.values):
    axes[1].text(v + 0.005, i, f'{v:.2%}', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-11-ethics-comms/ch40-responsible-ai/diagrams/adult_income_bias.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ adult_income_bias.png created")

# ============================================================================
# DIAGRAM 3: Fairness Trade-off
# ============================================================================
print("\n3. Generating fairness_tradeoff.png...")

# Prepare features for modeling
le_education = LabelEncoder()
le_race = LabelEncoder()

X = df_clean.copy()
X['education_encoded'] = le_education.fit_transform(X['education'])
X['race_encoded'] = le_race.fit_transform(X['race'])

feature_cols = ['age', 'hours-per-week', 'education_encoded', 'race_encoded']
X_model = X[feature_cols]
y = X['income_binary']
sensitive_feature = X['sex']

# Split data
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X_model, y, sensitive_feature, test_size=0.2, random_state=42, stratify=y
)

# Train baseline model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate baseline metrics
accuracy = accuracy_score(y_test, y_pred)
female_mask = (sensitive_test == 'Female')
male_mask = (sensitive_test == 'Male')

positive_rate_female = y_pred[female_mask].mean()
positive_rate_male = y_pred[male_mask].mean()

female_qualified = female_mask & (y_test == 1)
male_qualified = male_mask & (y_test == 1)
tpr_female = y_pred[female_qualified].mean() if female_qualified.sum() > 0 else 0
tpr_male = y_pred[male_qualified].mean() if male_qualified.sum() > 0 else 0

# Apply debiasing (threshold optimization)
def find_threshold_for_tpr(probabilities, true_labels, target_tpr=0.65):
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

proba_female = y_proba[female_mask]
proba_male = y_proba[male_mask]
y_test_female = y_test[female_mask]
y_test_male = y_test[male_mask]

target_tpr = 0.65
threshold_female = find_threshold_for_tpr(proba_female, y_test_female, target_tpr)
threshold_male = find_threshold_for_tpr(proba_male, y_test_male, target_tpr)

y_pred_fair = np.zeros_like(y_pred)
y_pred_fair[female_mask] = (proba_female >= threshold_female).astype(int)
y_pred_fair[male_mask] = (proba_male >= threshold_male).astype(int)

# Calculate fair metrics
accuracy_fair = accuracy_score(y_test, y_pred_fair)
positive_rate_female_fair = y_pred_fair[female_mask].mean()
positive_rate_male_fair = y_pred_fair[male_mask].mean()

tpr_female_fair = y_pred_fair[female_qualified].mean() if female_qualified.sum() > 0 else 0
tpr_male_fair = y_pred_fair[male_qualified].mean() if male_qualified.sum() > 0 else 0

# Create comparison visualization
metrics_comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Demo. Parity Diff', 'TPR Difference'],
    'Baseline': [accuracy, abs(positive_rate_male - positive_rate_female), abs(tpr_male - tpr_female)],
    'After Debiasing': [accuracy_fair, abs(positive_rate_male_fair - positive_rate_female_fair), abs(tpr_male_fair - tpr_female_fair)]
})

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics_comparison))
width = 0.35

bars1 = ax.bar(x - width/2, metrics_comparison['Baseline'], width,
               label='Baseline', color=COLORS['red'], edgecolor='black', linewidth=1.5, alpha=0.7)
bars2 = ax.bar(x + width/2, metrics_comparison['After Debiasing'], width,
               label='After Debiasing', color=COLORS['green'], edgecolor='black', linewidth=1.5, alpha=0.7)

ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
ax.set_title('Accuracy-Fairness Trade-off: Baseline vs. Debiased Model', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_comparison['Metric'], fontsize=11)
ax.legend(fontsize=11)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-11-ethics-comms/ch40-responsible-ai/diagrams/fairness_tradeoff.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ fairness_tradeoff.png created")

# ============================================================================
# DIAGRAM 4: Differential Privacy
# ============================================================================
print("\n4. Generating differential_privacy.png...")

# Generate synthetic salary data
n_employees = 500
true_salaries = np.random.normal(loc=75000, scale=15000, size=n_employees)
true_mean_salary = true_salaries.mean()

# Laplace mechanism
def laplace_mechanism(true_value, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale)
    return true_value + noise

# Sensitivity calculation
salary_range = 200000 - 20000
sensitivity = salary_range / n_employees

# Test different epsilon values
epsilons = [0.1, 1.0, 10.0]
n_queries = 100

results = {}
for epsilon in epsilons:
    noisy_means = [laplace_mechanism(true_mean_salary, sensitivity, epsilon)
                   for _ in range(n_queries)]
    results[epsilon] = noisy_means

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, epsilon in enumerate(epsilons):
    ax = axes[i]
    privacy_level = 'Strong' if epsilon < 1 else 'Moderate' if epsilon < 5 else 'Weak'

    ax.hist(results[epsilon], bins=30, alpha=0.7, color=COLORS['blue'], edgecolor='black', linewidth=1)
    ax.axvline(true_mean_salary, color=COLORS['red'], linestyle='--', linewidth=2.5,
               label=f'True mean: ${true_mean_salary:,.0f}')
    ax.set_xlabel('Noisy Mean Salary ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (out of 100 queries)', fontsize=12, fontweight='bold')
    ax.set_title(f'ε = {epsilon}\n({privacy_level} Privacy)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-11-ethics-comms/ch40-responsible-ai/diagrams/differential_privacy.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ differential_privacy.png created")

# ============================================================================
# DIAGRAM 5: DP Count Trade-off
# ============================================================================
print("\n5. Generating dp_count_tradeoff.png...")

# Generate synthetic dataset
n = 500
prevalence = 0.3
data = np.random.choice([0, 1], size=n, p=[1-prevalence, prevalence])
true_count = data.sum()

def dp_count(data, epsilon):
    true_count = data.sum()
    sensitivity = 1
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale)
    return true_count + noise

# Run experiments
epsilons = [0.1, 1.0, 10.0]
n_trials = 100
results = {}

for eps in epsilons:
    noisy_counts = [dp_count(data, eps) for _ in range(n_trials)]
    results[eps] = noisy_counts

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, eps in enumerate(epsilons):
    ax = axes[i]
    privacy_level = 'Strong' if eps < 1 else 'Moderate' if eps < 5 else 'Weak'

    ax.hist(results[eps], bins=30, alpha=0.7, color=COLORS['blue'], edgecolor='black', linewidth=1)
    ax.axvline(true_count, color=COLORS['red'], linestyle='--', linewidth=2.5,
               label=f'True count: {true_count}')
    ax.set_xlabel('Noisy Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'ε = {eps}\n({privacy_level} Privacy)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-11-ethics-comms/ch40-responsible-ai/diagrams/dp_count_tradeoff.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ dp_count_tradeoff.png created")

print("\n" + "="*70)
print("All diagrams generated successfully!")
print("="*70)
