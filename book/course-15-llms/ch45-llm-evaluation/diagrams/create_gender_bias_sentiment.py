import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Simulate sentiment scores for demonstration
# Male: more positive scores
male_scores = np.random.normal(0.75, 0.18, 5)
# Female: more negative scores (showing bias)
female_scores = np.random.normal(-0.42, 0.29, 5)

male_mean = np.mean(male_scores)
female_mean = np.mean(female_scores)

# Visualize sentiment distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Box plot comparison
box_data = [male_scores, female_scores]
bp = ax1.boxplot(box_data, labels=['Male\nPronouns', 'Female\nPronouns'], patch_artist=True,
            boxprops=dict(facecolor='#2196F3', alpha=0.7))
ax1.set_ylabel('Sentiment Score', fontsize=12)
ax1.set_title('Sentiment Distribution by Gender', fontsize=13, fontweight='bold')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.grid(True, alpha=0.3, axis='y')

# Bar chart of means with error bars
genders = ['Male', 'Female']
means = [male_mean, female_mean]
stds = [np.std(male_scores), np.std(female_scores)]

bars = ax2.bar(genders, means, yerr=stds, capsize=10, alpha=0.7,
               color=['#2196F3', '#F44336'])
ax2.set_ylabel('Mean Sentiment Score', fontsize=12)
ax2.set_title('Average Sentiment by Gender Pronoun', fontsize=13, fontweight='bold')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, mean in zip(bars, means):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-15/ch45/diagrams/gender_bias_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: gender_bias_sentiment.png")
