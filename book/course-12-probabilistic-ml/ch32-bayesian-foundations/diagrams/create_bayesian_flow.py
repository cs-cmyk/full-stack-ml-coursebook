#!/usr/bin/env python3
"""
Create Bayesian Inference Flow Diagram
Shows: Prior → Likelihood → Posterior with visual distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Set up the figure
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3,
                      left=0.08, right=0.95, top=0.92, bottom=0.08)

# Color palette
COLOR_PRIOR = '#2196F3'      # blue
COLOR_LIKELIHOOD = '#4CAF50' # green
COLOR_POSTERIOR = '#9C27B0'  # purple
COLOR_TEXT = '#333333'

# ============================================================================
# TOP ROW: Flow diagram with annotations
# ============================================================================
ax_flow = fig.add_subplot(gs[0, :])
ax_flow.set_xlim(0, 10)
ax_flow.set_ylim(0, 3)
ax_flow.axis('off')

# Box positions
box_width = 1.8
box_height = 0.8
y_center = 1.5

# Prior box
prior_x = 0.5
rect_prior = FancyBboxPatch((prior_x, y_center - box_height/2), box_width, box_height,
                            boxstyle="round,pad=0.1",
                            edgecolor=COLOR_PRIOR, facecolor=COLOR_PRIOR,
                            linewidth=3, alpha=0.7)
ax_flow.add_patch(rect_prior)
ax_flow.text(prior_x + box_width/2, y_center, 'Prior\nP(θ)',
             ha='center', va='center', fontsize=14, fontweight='bold', color='white')

# Plus sign
ax_flow.text(prior_x + box_width + 0.35, y_center, '×',
             ha='center', va='center', fontsize=24, fontweight='bold', color=COLOR_TEXT)

# Data/Likelihood box
data_x = prior_x + box_width + 0.7
rect_data = FancyBboxPatch((data_x, y_center - box_height/2), box_width, box_height,
                           boxstyle="round,pad=0.1",
                           edgecolor=COLOR_LIKELIHOOD, facecolor=COLOR_LIKELIHOOD,
                           linewidth=3, alpha=0.7)
ax_flow.add_patch(rect_data)
ax_flow.text(data_x + box_width/2, y_center, 'Likelihood\nP(D|θ)',
             ha='center', va='center', fontsize=14, fontweight='bold', color='white')

# Arrow with "Bayes' Theorem"
arrow_x = data_x + box_width + 0.2
arrow = FancyArrowPatch((arrow_x, y_center), (arrow_x + 1.5, y_center),
                       arrowstyle='->', mutation_scale=30, linewidth=3,
                       color=COLOR_POSTERIOR)
ax_flow.add_patch(arrow)
ax_flow.text(arrow_x + 0.75, y_center + 0.5, "Bayes' Theorem",
             ha='center', va='bottom', fontsize=12, fontweight='bold',
             color=COLOR_POSTERIOR, style='italic')

# Posterior box
post_x = arrow_x + 1.7
rect_post = FancyBboxPatch((post_x, y_center - box_height/2), box_width, box_height,
                          boxstyle="round,pad=0.1",
                          edgecolor=COLOR_POSTERIOR, facecolor=COLOR_POSTERIOR,
                          linewidth=3, alpha=0.7)
ax_flow.add_patch(rect_post)
ax_flow.text(post_x + box_width/2, y_center, 'Posterior\nP(θ|D)',
             ha='center', va='center', fontsize=14, fontweight='bold', color='white')

# Add annotations below boxes
ax_flow.text(prior_x + box_width/2, y_center - 0.7,
             'Initial beliefs\nbefore data',
             ha='center', va='top', fontsize=10, color=COLOR_TEXT, style='italic')

ax_flow.text(data_x + box_width/2, y_center - 0.7,
             'Evidence from\nobserved data',
             ha='center', va='top', fontsize=10, color=COLOR_TEXT, style='italic')

ax_flow.text(post_x + box_width/2, y_center - 0.7,
             'Updated beliefs\nafter data',
             ha='center', va='top', fontsize=10, color=COLOR_TEXT, style='italic')

# Title
ax_flow.text(5, 2.7, 'Bayesian Inference: Prior + Likelihood → Posterior',
             ha='center', va='top', fontsize=16, fontweight='bold', color=COLOR_TEXT)

# ============================================================================
# BOTTOM ROW: Distribution plots
# ============================================================================

# Example: Beta-Binomial with 5 successes out of 10 trials
# Prior: Beta(2, 2) - weakly informative
# Data: 5 successes, 5 failures
# Posterior: Beta(2+5, 2+5) = Beta(7, 7)

p_values = np.linspace(0, 1, 500)

# Prior distribution
alpha_prior, beta_prior = 2, 2
prior_dist = beta(alpha_prior, beta_prior)
prior_density = prior_dist.pdf(p_values)

# Likelihood (scaled for visualization)
k, n = 5, 10
likelihood = p_values**k * (1 - p_values)**(n - k)
likelihood_scaled = likelihood / np.max(likelihood) * np.max(prior_density) * 1.2

# Posterior distribution
alpha_post = alpha_prior + k
beta_post = beta_prior + (n - k)
posterior_dist = beta(alpha_post, beta_post)
posterior_density = posterior_dist.pdf(p_values)

# Plot 1: Prior
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(p_values, prior_density, color=COLOR_PRIOR, linewidth=3)
ax1.fill_between(p_values, 0, prior_density, alpha=0.3, color=COLOR_PRIOR)
ax1.set_xlabel('Parameter θ', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Prior: P(θ)', fontsize=12, fontweight='bold', color=COLOR_PRIOR)
ax1.grid(alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_ylim(0, max(prior_density) * 1.2)
ax1.text(0.5, max(prior_density) * 1.05, 'Initial belief:\nθ ~ Beta(2,2)',
         ha='center', va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Plot 2: Likelihood
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(p_values, likelihood_scaled, color=COLOR_LIKELIHOOD, linewidth=3, linestyle='--')
ax2.fill_between(p_values, 0, likelihood_scaled, alpha=0.3, color=COLOR_LIKELIHOOD)
ax2.set_xlabel('Parameter θ', fontsize=11)
ax2.set_ylabel('Scaled Likelihood', fontsize=11)
ax2.set_title('Likelihood: P(D|θ)', fontsize=12, fontweight='bold', color=COLOR_LIKELIHOOD)
ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_ylim(0, max(prior_density) * 1.2)
ax2.text(0.5, max(likelihood_scaled) * 0.95, 'Observed data:\n5 successes,\n5 failures',
         ha='center', va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Plot 3: Posterior
ax3 = fig.add_subplot(gs[1, 2])
# Show all three for comparison
ax3.plot(p_values, prior_density, color=COLOR_PRIOR, linewidth=2,
         alpha=0.5, label='Prior', linestyle=':')
ax3.plot(p_values, likelihood_scaled, color=COLOR_LIKELIHOOD, linewidth=2,
         alpha=0.5, label='Likelihood (scaled)', linestyle='--')
ax3.plot(p_values, posterior_density, color=COLOR_POSTERIOR, linewidth=3,
         label='Posterior')
ax3.fill_between(p_values, 0, posterior_density, alpha=0.3, color=COLOR_POSTERIOR)
ax3.axvline(posterior_dist.mean(), color=COLOR_POSTERIOR, linestyle='--',
            linewidth=1.5, alpha=0.7, label=f'Mean = {posterior_dist.mean():.2f}')
ax3.set_xlabel('Parameter θ', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title('Posterior: P(θ|D)', fontsize=12, fontweight='bold', color=COLOR_POSTERIOR)
ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax3.grid(alpha=0.3, linestyle='--', linewidth=0.5)
ax3.set_ylim(0, max(posterior_density) * 1.2)

# Add credible interval
ci_lower = posterior_dist.ppf(0.025)
ci_upper = posterior_dist.ppf(0.975)
ax3.fill_between(p_values, 0, posterior_density,
                where=(p_values >= ci_lower) & (p_values <= ci_upper),
                alpha=0.4, color=COLOR_POSTERIOR,
                label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
ax3.text(0.5, max(posterior_density) * 1.05,
         f'Updated belief:\nθ ~ Beta({alpha_post},{beta_post})',
         ha='center', va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-12/ch32/diagrams/bayesian_flow.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Created bayesian_flow.png")
