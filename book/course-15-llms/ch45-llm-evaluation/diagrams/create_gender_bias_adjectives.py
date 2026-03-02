import matplotlib.pyplot as plt
import numpy as np

# Sample data showing stereotypical adjectives for demonstration
male_adjectives = ['Confident', 'Strong', 'Assertive', 'Analytical', 'Decisive']
male_counts = [45, 38, 35, 32, 28]

female_adjectives = ['Supportive', 'Caring', 'Collaborative', 'Organized', 'Detail-oriented']
female_counts = [42, 39, 36, 33, 30]

# Create side-by-side bar charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Male adjectives
ax1.barh(male_adjectives, male_counts, color='#2196F3', alpha=0.7)
ax1.set_xlabel('Frequency', fontsize=12)
ax1.set_title('Top Adjectives: Male Pronouns', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

# Female adjectives
ax2.barh(female_adjectives, female_counts, color='#F44336', alpha=0.7)
ax2.set_xlabel('Frequency', fontsize=12)
ax2.set_title('Top Adjectives: Female Pronouns', fontsize=13, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-15/ch45/diagrams/gender_bias_adjectives.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: gender_bias_adjectives.png")
