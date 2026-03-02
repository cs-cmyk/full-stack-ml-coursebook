import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set color palette
BLUE = '#2196F3'
GREEN = '#4CAF50'
ORANGE = '#FF9800'
RED = '#F44336'
PURPLE = '#9C27B0'
GRAY = '#607D8B'

# Create figure with subplots
fig = plt.figure(figsize=(14, 10))

# 1. Linear vs Cyclical Encoding
ax1 = plt.subplot(2, 3, 1)
hours_linear = np.arange(24)
ax1.plot(hours_linear, hours_linear, 'o-', linewidth=2, markersize=8, color=BLUE)
ax1.axhline(y=0, color=RED, linestyle='--', linewidth=2, alpha=0.7)
ax1.axhline(y=23, color=RED, linestyle='--', linewidth=2, alpha=0.7)
ax1.text(12, 11.5, 'Distance = 23', fontsize=12, ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax1.set_xlabel('Hour (Linear)', fontsize=12)
ax1.set_ylabel('Numeric Value', fontsize=12)
ax1.set_title('A. Linear Encoding Problem', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Cyclical Encoding Circle
ax2 = plt.subplot(2, 3, 2, projection='polar')
hours = np.arange(24)
angles = 2 * np.pi * hours / 24
colors = plt.cm.twilight(hours / 24)
ax2.scatter(angles, np.ones(24), c=colors, s=100, zorder=3)
for i, hour in enumerate([0, 6, 12, 18, 23]):
    idx = np.where(hours == hour)[0][0]
    ax2.annotate(f'{hour}h', xy=(angles[idx], 1), xytext=(angles[idx], 1.15),
                ha='center', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 1.3)
ax2.set_yticks([])
ax2.set_xticks([])
ax2.set_title('B. Cyclical Encoding (Circle)', fontsize=13, fontweight='bold', pad=20)

# 3. Sine and Cosine Waves
ax3 = plt.subplot(2, 3, 3)
hours_extended = np.linspace(0, 48, 200)
hour_sin = np.sin(2 * np.pi * hours_extended / 24)
hour_cos = np.cos(2 * np.pi * hours_extended / 24)
ax3.plot(hours_extended, hour_sin, label='sin(2π·h/24)', linewidth=2, color=ORANGE)
ax3.plot(hours_extended, hour_cos, label='cos(2π·h/24)', linewidth=2, color=GREEN)
ax3.axvline(x=0, color=RED, linestyle='--', alpha=0.5)
ax3.axvline(x=23, color=RED, linestyle='--', alpha=0.5)
ax3.text(11.5, 0.5, 'Hour 0 and 23\nare close!', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax3.set_xlabel('Hour', fontsize=12)
ax3.set_ylabel('Feature Value', fontsize=12)
ax3.set_title('C. Sine-Cosine Transformation', fontsize=13, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Lag Features Timeline
ax4 = plt.subplot(2, 3, 4)
days = np.arange(20)
np.random.seed(42)
values = 50 + 10*np.sin(2*np.pi*days/7) + np.random.normal(0, 2, 20)
ax4.plot(days, values, 'o-', label='Original', linewidth=2, markersize=6, color=BLUE)
ax4.plot(days, np.roll(values, 1), 'o--', label='Lag-1 (shift 1)',
         linewidth=2, markersize=5, alpha=0.7, color=GREEN)
ax4.plot(days, np.roll(values, 7), 's--', label='Lag-7 (shift 7)',
         linewidth=2, markersize=5, alpha=0.7, color=ORANGE)
# Highlight prediction point
ax4.axvline(x=10, color=RED, linestyle=':', linewidth=2)
ax4.text(10.5, 65, 'Predict\nhere', fontsize=11, color=RED, fontweight='bold')
ax4.set_xlabel('Day', fontsize=12)
ax4.set_ylabel('Value', fontsize=12)
ax4.set_title('D. Lag Features', fontsize=13, fontweight='bold')
ax4.legend(loc='upper left', fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. Rolling Window
ax5 = plt.subplot(2, 3, 5)
days2 = np.arange(30)
np.random.seed(42)
values2 = 50 + 15*np.sin(2*np.pi*days2/10) + np.random.normal(0, 5, 30)
rolling_7 = np.convolve(values2, np.ones(7)/7, mode='same')
ax5.plot(days2, values2, 'o-', label='Original (noisy)', linewidth=1.5,
         markersize=4, alpha=0.6, color=GRAY)
ax5.plot(days2, rolling_7, linewidth=3, label='Rolling 7-day mean', color=BLUE)
# Show window
window_start = 15
ax5.axvspan(window_start-3, window_start+3, alpha=0.2, color='yellow')
ax5.text(window_start, 25, 'Window', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax5.set_xlabel('Day', fontsize=12)
ax5.set_ylabel('Value', fontsize=12)
ax5.set_title('E. Rolling Window Features', fontsize=13, fontweight='bold')
ax5.legend(loc='upper right', fontsize=10)
ax5.grid(True, alpha=0.3)

# 6. Temporal Train/Test Split
ax6 = plt.subplot(2, 3, 6)
ax6.barh([1], [75], left=[0], height=0.3, color=GREEN, alpha=0.7, label='Training (past)')
ax6.barh([1], [25], left=[75], height=0.3, color=ORANGE, alpha=0.7, label='Test (future)')
ax6.barh([0], [50], left=[0], height=0.3, color=RED, alpha=0.3)
ax6.barh([0], [50], left=[50], height=0.3, color=RED, alpha=0.3)
ax6.text(50, 0, '✗ RANDOM SPLIT\n(leakage!)', ha='center', va='center',
         fontsize=12, fontweight='bold', color='darkred')
ax6.text(37.5, 1, 'TRAIN', ha='center', va='center', fontsize=12, fontweight='bold')
ax6.text(87.5, 1, 'TEST', ha='center', va='center', fontsize=12, fontweight='bold')
ax6.arrow(37.5, 1.5, 45, 0, head_width=0.1, head_length=3, fc='black', ec='black')
ax6.text(60, 1.7, 'Time flows →', ha='center', fontsize=11)
ax6.set_xlim(0, 100)
ax6.set_ylim(-0.5, 2)
ax6.set_xlabel('Data Timeline (%)', fontsize=12)
ax6.set_yticks([0, 1])
ax6.set_yticklabels(['Wrong', 'Correct'])
ax6.set_title('F. Temporal Split (No Leakage)', fontsize=13, fontweight='bold')
ax6.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('time_features_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: time_features_overview.png")
