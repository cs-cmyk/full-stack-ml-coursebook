#!/usr/bin/env python3
"""
Create probability_concepts.png - 4-panel visualization showing:
- Panel A: Sample Space with Venn diagram
- Panel B: Conditional Probability tree diagram
- Panel C: Bayes' Theorem flow
- Panel D: Common Distributions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch
import numpy as np
from scipy import stats

# Set up the figure with 4 panels
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Color palette
BLUE = '#2196F3'
GREEN = '#4CAF50'
ORANGE = '#FF9800'
RED = '#F44336'
PURPLE = '#9C27B0'
GRAY = '#607D8B'

# ============================================================
# PANEL A: Sample Space and Venn Diagram
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_aspect('equal')
ax1.axis('off')

# Draw sample space rectangle
sample_space = Rectangle((0.5, 0.5), 9, 9, linewidth=3,
                         edgecolor='black', facecolor='white')
ax1.add_patch(sample_space)
ax1.text(5, 9.8, 'Sample Space Ω', ha='center', va='top',
         fontsize=14, fontweight='bold')

# Draw two overlapping circles for events A and B
circle_a = Circle((4, 5), 2.5, alpha=0.3, facecolor=BLUE,
                 edgecolor=BLUE, linewidth=2, label='Event A')
circle_b = Circle((6, 5), 2.5, alpha=0.3, facecolor=GREEN,
                 edgecolor=GREEN, linewidth=2, label='Event B')
ax1.add_patch(circle_a)
ax1.add_patch(circle_b)

# Labels
ax1.text(3, 6.5, 'A', fontsize=16, fontweight='bold', color=BLUE)
ax1.text(7, 6.5, 'B', fontsize=16, fontweight='bold', color=GREEN)
ax1.text(5, 5, 'A ∩ B', fontsize=12, fontweight='bold', ha='center')
ax1.text(1.5, 8, 'A ∪ B = Union\nA ∩ B = Intersection',
         fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax1.text(5, 0.2, 'Panel A: Sample Space & Events', ha='center',
         fontsize=13, fontweight='bold')

# ============================================================
# PANEL B: Conditional Probability Tree Diagram
# ============================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Title
ax2.text(5, 9.5, 'Panel B: Conditional Probability', ha='center',
         fontsize=13, fontweight='bold')

# Tree structure
# Root
ax2.plot([2, 2], [7.5, 7.5], 'ko', markersize=10)
ax2.text(1, 7.5, 'Start', fontsize=12, ha='right', va='center', fontweight='bold')

# First level - Event B or not B
ax2.plot([2, 4], [7.5, 6], 'k-', linewidth=2)
ax2.plot([2, 4], [7.5, 9], 'k-', linewidth=2, linestyle='--', alpha=0.5)
ax2.plot([4, 4], [6, 6], 'o', color=GREEN, markersize=12)
ax2.plot([4, 4], [9, 9], 'o', color=GRAY, markersize=10, alpha=0.5)

ax2.text(3, 6.5, 'P(B)', fontsize=11, ha='center', color=GREEN, fontweight='bold')
ax2.text(3, 8.5, 'P(B̄)', fontsize=11, ha='center', color=GRAY, alpha=0.5)

# Second level - Event A given B
ax2.plot([4, 6.5], [6, 5], 'k-', linewidth=2, color=BLUE)
ax2.plot([4, 6.5], [6, 7], 'k-', linewidth=2, color=GRAY, alpha=0.6)
ax2.plot([6.5, 6.5], [5, 5], 'o', color=BLUE, markersize=12)
ax2.plot([6.5, 6.5], [7, 7], 'o', color=GRAY, markersize=10, alpha=0.6)

ax2.text(5.2, 5.3, 'P(A|B)', fontsize=11, ha='center', color=BLUE, fontweight='bold')
ax2.text(5.2, 6.7, 'P(Ā|B)', fontsize=11, ha='center', color=GRAY, alpha=0.6)

# Result labels
ax2.text(7.5, 5, 'A ∩ B', fontsize=11, ha='left', va='center',
         bbox=dict(boxstyle='round', facecolor=BLUE, alpha=0.3))
ax2.text(7.5, 7, 'Ā ∩ B', fontsize=11, ha='left', va='center',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# Formula box
formula_text = 'P(A|B) = P(A ∩ B) / P(B)'
ax2.text(5, 2.5, formula_text, fontsize=13, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
         fontweight='bold')
ax2.text(5, 1.5, '"Probability of A given B"', fontsize=11,
         ha='center', style='italic')

# ============================================================
# PANEL C: Bayes' Theorem Flow
# ============================================================
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')

# Title
ax3.text(5, 9.5, "Panel C: Bayes' Theorem", ha='center',
         fontsize=13, fontweight='bold')

# Flow diagram: Prior -> Evidence -> Posterior
# Prior
prior_box = FancyBboxPatch((0.5, 6.5), 2, 1.2, boxstyle="round,pad=0.1",
                           edgecolor=PURPLE, facecolor=PURPLE, alpha=0.3, linewidth=2)
ax3.add_patch(prior_box)
ax3.text(1.5, 7.1, 'Prior\nP(H)', fontsize=11, ha='center', va='center', fontweight='bold')

# Arrow 1
arrow1 = FancyArrowPatch((2.5, 7.1), (3.8, 7.1),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax3.add_patch(arrow1)

# Evidence
evidence_box = FancyBboxPatch((3.8, 6.5), 2.4, 1.2, boxstyle="round,pad=0.1",
                             edgecolor=ORANGE, facecolor=ORANGE, alpha=0.3, linewidth=2)
ax3.add_patch(evidence_box)
ax3.text(5, 7.1, 'Evidence\nE arrives', fontsize=11, ha='center', va='center', fontweight='bold')

# Arrow 2
arrow2 = FancyArrowPatch((6.2, 7.1), (7.5, 7.1),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax3.add_patch(arrow2)

# Posterior
posterior_box = FancyBboxPatch((7.5, 6.5), 2, 1.2, boxstyle="round,pad=0.1",
                              edgecolor=GREEN, facecolor=GREEN, alpha=0.3, linewidth=2)
ax3.add_patch(posterior_box)
ax3.text(8.5, 7.1, 'Posterior\nP(H|E)', fontsize=11, ha='center', va='center', fontweight='bold')

# Bayes' formula - main box
formula_box = FancyBboxPatch((1, 3.8), 8, 1.8, boxstyle="round,pad=0.15",
                            edgecolor='black', facecolor='lightyellow', linewidth=2)
ax3.add_patch(formula_box)

# Formula text
ax3.text(5, 5.2, "P(H|E) = P(E|H) × P(H) / P(E)", fontsize=14, ha='center',
         fontweight='bold', family='monospace')
ax3.text(5, 4.3, "Posterior = Likelihood × Prior / Evidence", fontsize=11,
         ha='center', style='italic')

# Component labels with colors
ax3.text(5, 2.5, 'P(H) = prior belief', fontsize=10, ha='center', color=PURPLE)
ax3.text(5, 2, 'P(E|H) = likelihood of evidence given hypothesis', fontsize=10, ha='center', color=BLUE)
ax3.text(5, 1.5, 'P(E) = total probability of evidence', fontsize=10, ha='center', color=GRAY)
ax3.text(5, 1, 'P(H|E) = updated belief after seeing evidence', fontsize=10, ha='center', color=GREEN)

# ============================================================
# PANEL D: Common Distributions
# ============================================================
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

# Title
ax4.text(5, 9.5, 'Panel D: Common Distributions', ha='center',
         fontsize=13, fontweight='bold')

# Create 4 mini-plots for distributions
# Bernoulli
ax_bern = fig.add_axes([0.52, 0.31, 0.18, 0.12])
x_bern = [0, 1]
y_bern = [0.6, 0.4]
ax_bern.bar(x_bern, y_bern, color=BLUE, alpha=0.7, edgecolor='black', width=0.4)
ax_bern.set_xlim(-0.5, 1.5)
ax_bern.set_ylim(0, 0.7)
ax_bern.set_xticks([0, 1])
ax_bern.set_xticklabels(['0', '1'], fontsize=9)
ax_bern.set_ylabel('P(X)', fontsize=9)
ax_bern.set_title('Bernoulli(p=0.4)', fontsize=10, fontweight='bold')
ax_bern.grid(True, alpha=0.3)

# Binomial
ax_binom = fig.add_axes([0.75, 0.31, 0.18, 0.12])
x_binom = np.arange(0, 11)
y_binom = stats.binom.pmf(x_binom, n=10, p=0.5)
ax_binom.bar(x_binom, y_binom, color=GREEN, alpha=0.7, edgecolor='black', width=0.6)
ax_binom.set_xlim(-0.5, 10.5)
ax_binom.set_xticks([0, 5, 10])
ax_binom.set_xticklabels(['0', '5', '10'], fontsize=9)
ax_binom.set_ylabel('P(X)', fontsize=9)
ax_binom.set_title('Binomial(n=10, p=0.5)', fontsize=10, fontweight='bold')
ax_binom.grid(True, alpha=0.3)

# Uniform
ax_unif = fig.add_axes([0.52, 0.06, 0.18, 0.12])
x_unif = np.linspace(0, 10, 100)
y_unif = stats.uniform.pdf(x_unif, loc=2, scale=6)
ax_unif.fill_between(x_unif, y_unif, color=ORANGE, alpha=0.7, edgecolor='black', linewidth=1.5)
ax_unif.set_xlim(0, 10)
ax_unif.set_ylim(0, 0.2)
ax_unif.set_xticks([2, 5, 8])
ax_unif.set_xticklabels(['a', 'x', 'b'], fontsize=9)
ax_unif.set_ylabel('f(x)', fontsize=9)
ax_unif.set_title('Uniform(a, b)', fontsize=10, fontweight='bold')
ax_unif.grid(True, alpha=0.3)

# Normal
ax_norm = fig.add_axes([0.75, 0.06, 0.18, 0.12])
x_norm = np.linspace(-4, 4, 200)
y_norm = stats.norm.pdf(x_norm, loc=0, scale=1)
ax_norm.fill_between(x_norm, y_norm, color=RED, alpha=0.7, edgecolor='black', linewidth=1.5)
ax_norm.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax_norm.axvline(-1, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax_norm.axvline(1, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax_norm.set_xlim(-4, 4)
ax_norm.set_ylim(0, 0.45)
ax_norm.set_xticks([-1, 0, 1])
ax_norm.set_xticklabels(['μ-σ', 'μ', 'μ+σ'], fontsize=9)
ax_norm.set_ylabel('f(x)', fontsize=9)
ax_norm.set_title('Normal(μ, σ²)', fontsize=10, fontweight='bold')
ax_norm.grid(True, alpha=0.3)

# Save figure
plt.savefig('/home/chirag/ds-book/book/course-01-foundations/ch03-probability/diagrams/probability_concepts.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Created: probability_concepts.png")
