"""
Diagram 4: Encoding in ML Pipeline (Avoiding Data Leakage)
Flowchart showing proper pipeline to prevent data leakage
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Color palette
blue = '#2196F3'
green = '#4CAF50'
orange = '#FF9800'
red = '#F44336'
purple = '#9C27B0'
gray = '#607D8B'

# ============ TITLE ============
title_box = FancyBboxPatch((1, 9.2), 12, 0.6,
                          boxstyle="round,pad=0.1",
                          edgecolor=blue, facecolor=blue, linewidth=3)
ax.add_patch(title_box)
ax.text(7, 9.5, 'Proper ML Pipeline: Avoiding Data Leakage in Categorical Encoding',
       ha='center', va='center', fontsize=14, weight='bold', color='white')

# ============ STEP 1: RAW DATA ============
y_pos = 8.5
box1 = FancyBboxPatch((5.5, y_pos - 0.3), 3, 0.6,
                     boxstyle="round,pad=0.1",
                     edgecolor=gray, facecolor='#E0E0E0', linewidth=2)
ax.add_patch(box1)
ax.text(7, y_pos, 'Raw Dataset\n(X features, y target)',
       ha='center', va='center', fontsize=11, weight='bold')

# Arrow down
arrow1 = FancyArrowPatch((7, y_pos - 0.4), (7, y_pos - 0.9),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=3, color=gray)
ax.add_patch(arrow1)

# ============ STEP 2: TRAIN/TEST SPLIT ============
y_pos = 7.2
split_box = FancyBboxPatch((4.5, y_pos - 0.4), 5, 0.8,
                          boxstyle="round,pad=0.1",
                          edgecolor=orange, facecolor='#FFF3E0', linewidth=3)
ax.add_patch(split_box)
ax.text(7, y_pos + 0.2, 'Train/Test Split',
       ha='center', va='center', fontsize=12, weight='bold', color=orange)
ax.text(7, y_pos - 0.15, 'CRITICAL: Split BEFORE encoding!',
       ha='center', va='center', fontsize=9, style='italic', color=orange)

# Warning icon
warning_circle = Circle((4.8, y_pos), 0.2, edgecolor=red, facecolor='#FFEBEE', linewidth=2)
ax.add_patch(warning_circle)
ax.text(4.8, y_pos, '⚠', ha='center', va='center', fontsize=16, color=red)

# Split into two branches
arrow_left = FancyArrowPatch((6, y_pos - 0.5), (3, y_pos - 1.2),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=3, color=blue)
ax.add_patch(arrow_left)
ax.text(4.2, y_pos - 0.9, 'Train Set (80%)', fontsize=9, weight='bold', color=blue)

arrow_right = FancyArrowPatch((8, y_pos - 0.5), (11, y_pos - 1.2),
                             arrowstyle='->', mutation_scale=25,
                             linewidth=3, color=green)
ax.add_patch(arrow_right)
ax.text(9.8, y_pos - 0.9, 'Test Set (20%)', fontsize=9, weight='bold', color=green)

# ============ LEFT BRANCH: TRAINING PIPELINE ============
y_train = 5.5

# Train data
train_box = FancyBboxPatch((1.5, y_train - 0.25), 3, 0.5,
                          boxstyle="round,pad=0.1",
                          edgecolor=blue, facecolor='#E3F2FD', linewidth=2)
ax.add_patch(train_box)
ax.text(3, y_train, 'X_train, y_train',
       ha='center', va='center', fontsize=10, weight='bold', color=blue)

# Arrow down
arrow_train1 = FancyArrowPatch((3, y_train - 0.35), (3, y_train - 0.8),
                              arrowstyle='->', mutation_scale=20,
                              linewidth=2, color=blue)
ax.add_patch(arrow_train1)

# Fit encoder
y_train2 = 4.5
fit_box = FancyBboxPatch((1.2, y_train2 - 0.35), 3.6, 0.7,
                        boxstyle="round,pad=0.1",
                        edgecolor=blue, facecolor=blue, linewidth=2, alpha=0.7)
ax.add_patch(fit_box)
ax.text(3, y_train2 + 0.1, 'FIT Encoder',
       ha='center', va='center', fontsize=11, weight='bold', color='white')
ax.text(3, y_train2 - 0.15, 'Learn encoding parameters\nfrom training data ONLY',
       ha='center', va='center', fontsize=8, color='white')

# Arrow down
arrow_train2 = FancyArrowPatch((3, y_train2 - 0.45), (3, y_train2 - 0.9),
                              arrowstyle='->', mutation_scale=20,
                              linewidth=2, color=blue)
ax.add_patch(arrow_train2)

# Transform train
y_train3 = 3.3
transform_train_box = FancyBboxPatch((1.5, y_train3 - 0.25), 3, 0.5,
                                    boxstyle="round,pad=0.1",
                                    edgecolor=blue, facecolor='#E3F2FD', linewidth=2)
ax.add_patch(transform_train_box)
ax.text(3, y_train3, 'TRANSFORM X_train\n(using fitted encoder)',
       ha='center', va='center', fontsize=9, weight='bold', color=blue)

# Arrow down
arrow_train3 = FancyArrowPatch((3, y_train3 - 0.35), (3, y_train3 - 0.8),
                              arrowstyle='->', mutation_scale=20,
                              linewidth=2, color=blue)
ax.add_patch(arrow_train3)

# Train model
y_train4 = 2.2
model_train_box = FancyBboxPatch((1.2, y_train4 - 0.35), 3.6, 0.7,
                                boxstyle="round,pad=0.1",
                                edgecolor=blue, facecolor=blue, linewidth=2, alpha=0.7)
ax.add_patch(model_train_box)
ax.text(3, y_train4 + 0.1, 'FIT Model',
       ha='center', va='center', fontsize=11, weight='bold', color='white')
ax.text(3, y_train4 - 0.15, 'Train on encoded features',
       ha='center', va='center', fontsize=8, color='white')

# ============ RIGHT BRANCH: TEST PIPELINE ============
# Test data
test_box = FancyBboxPatch((9.5, y_train - 0.25), 3, 0.5,
                         boxstyle="round,pad=0.1",
                         edgecolor=green, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(test_box)
ax.text(11, y_train, 'X_test, y_test',
       ha='center', va='center', fontsize=10, weight='bold', color=green)

# Wait annotation
ax.text(11, y_train - 0.6, '⏳ Wait for encoder\nto be fitted on train',
       ha='center', va='center', fontsize=8, style='italic', color=gray,
       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=gray, linewidth=1))

# Arrow showing encoder transfer
arrow_encoder = FancyArrowPatch((4.8, y_train2), (9.2, y_train2),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2.5, color=purple, linestyle='--')
ax.add_patch(arrow_encoder)
ax.text(7, y_train2 + 0.3, 'Use SAME fitted encoder', fontsize=9, weight='bold', color=purple)

# Arrow down from test
arrow_test1 = FancyArrowPatch((11, y_train - 0.35), (11, y_train3 + 0.35),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=2, color=green)
ax.add_patch(arrow_test1)

# Transform test (NO FIT!)
transform_test_box = FancyBboxPatch((9.5, y_train3 - 0.25), 3, 0.5,
                                   boxstyle="round,pad=0.1",
                                   edgecolor=green, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(transform_test_box)
ax.text(11, y_train3, 'TRANSFORM X_test\n(NO FIT - use learned params)',
       ha='center', va='center', fontsize=9, weight='bold', color=green)

# Critical note
critical_box = FancyBboxPatch((8.8, y_train3 - 0.8), 4.4, 0.4,
                             boxstyle="round,pad=0.05",
                             edgecolor=red, facecolor='#FFEBEE', linewidth=2)
ax.add_patch(critical_box)
ax.text(11, y_train3 - 0.6, '⚠ NEVER fit encoder on test data!\n(Causes data leakage)',
       ha='center', va='center', fontsize=8, weight='bold', color=red)

# Arrow down
arrow_test2 = FancyArrowPatch((11, y_train3 - 0.35), (11, y_train4 + 0.45),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=2, color=green)
ax.add_patch(arrow_test2)

# Arrow showing model transfer
arrow_model = FancyArrowPatch((4.8, y_train4), (9.2, y_train4),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color=purple, linestyle='--')
ax.add_patch(arrow_model)
ax.text(7, y_train4 + 0.3, 'Use trained model', fontsize=9, weight='bold', color=purple)

# Predict on test
predict_box = FancyBboxPatch((9.5, y_train4 - 0.35), 3, 0.7,
                            boxstyle="round,pad=0.1",
                            edgecolor=green, facecolor=green, linewidth=2, alpha=0.7)
ax.add_patch(predict_box)
ax.text(11, y_train4 + 0.1, 'PREDICT on X_test',
       ha='center', va='center', fontsize=11, weight='bold', color='white')
ax.text(11, y_train4 - 0.15, 'Evaluate performance',
       ha='center', va='center', fontsize=8, color='white')

# ============ BOTTOM: KEY PRINCIPLES ============
y_bottom = 0.8
principle_box = FancyBboxPatch((0.5, y_bottom - 0.5), 13, 0.9,
                              boxstyle="round,pad=0.1",
                              edgecolor=blue, facecolor='#E3F2FD', linewidth=2)
ax.add_patch(principle_box)

ax.text(7, y_bottom + 0.25, 'Key Principles to Prevent Data Leakage:',
       ha='center', va='center', fontsize=12, weight='bold', color=blue)

principles = [
    "1. Split data BEFORE any preprocessing or encoding",
    "2. FIT encoders (learn parameters) on training data only",
    "3. TRANSFORM both train and test using the fitted encoder",
    "4. Never let test set information influence encoder parameters"
]

for i, principle in enumerate(principles):
    ax.text(7, y_bottom - 0.05 - i * 0.15, principle,
           ha='center', va='center', fontsize=9, color=gray)

# ============ SKLEARN PIPELINE SHORTCUT ============
y_sklearn = -0.2
sklearn_box = FancyBboxPatch((0.5, y_sklearn - 0.4), 13, 0.7,
                            boxstyle="round,pad=0.1",
                            edgecolor=purple, facecolor='#F3E5F5', linewidth=2)
ax.add_patch(sklearn_box)

ax.text(7, y_sklearn + 0.15, '💡 Pro Tip: Use sklearn Pipeline to automate this pattern!',
       ha='center', va='center', fontsize=11, weight='bold', color=purple)
ax.text(7, y_sklearn - 0.1, 'Pipeline([("encoder", OneHotEncoder()), ("model", LogisticRegression())])',
       ha='center', va='center', fontsize=9, family='monospace', color=purple)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-03-eda-features/ch14-categorical-features/diagrams/04_pipeline_leakage.png',
           dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Diagram 4 saved: 04_pipeline_leakage.png")
