# Introduction to preference learning - foundation for RLHF
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.linear_model import LogisticRegression
from matplotlib.patches import Patch

np.random.seed(42)

class BradleyTerryPreferenceLearning:
    """
    Implements Bradley-Terry model for learning reward functions from pairwise preferences.
    This is the foundation of RLHF (Reinforcement Learning from Human Feedback).
    """

    def __init__(self):
        self.reward_model = None

    def generate_synthetic_preferences(self, n_pairs=100):
        """
        Generate synthetic preference data where humans prefer responses that are:
        - More informative (feature 0)
        - More concise (feature 1)
        - More accurate (feature 2)
        """
        preferences = []

        for _ in range(n_pairs):
            # Generate two responses with random features
            response_A = np.random.uniform(0, 1, 3)
            response_B = np.random.uniform(0, 1, 3)

            # True human reward function (unknown to the model)
            # Weights: informativeness=2.0, conciseness=1.5, accuracy=3.0
            true_weights = np.array([2.0, 1.5, 3.0])

            reward_A = np.dot(response_A, true_weights)
            reward_B = np.dot(response_B, true_weights)

            # Human prefers higher reward response (with some noise)
            prob_prefer_A = sigmoid(reward_A - reward_B)
            prefer_A = np.random.random() < prob_prefer_A

            # Store as training example
            # Features: concatenate both responses
            # Label: 1 if A preferred, 0 if B preferred
            features = np.concatenate([response_A, response_B])
            preferences.append((features, 1 if prefer_A else 0))

        X = np.array([p[0] for p in preferences])
        y = np.array([p[1] for p in preferences])

        return X, y, true_weights

    def train_reward_model(self, X, y):
        """
        Train Bradley-Terry model to predict preferences.
        Given features of two responses, predict which is preferred.
        """
        # Logistic regression predicts P(A preferred over B)
        # based on feature differences
        self.reward_model = LogisticRegression(random_state=42, max_iter=1000)
        self.reward_model.fit(X, y)

        return self.reward_model

# Generate preference dataset
bt_model = BradleyTerryPreferenceLearning()
X_prefs, y_prefs, true_weights = bt_model.generate_synthetic_preferences(n_pairs=200)

# Train reward model from preferences
trained_model = bt_model.train_reward_model(X_prefs, y_prefs)

# Extract learned weights
learned_weights = (trained_model.coef_[0][:3] - trained_model.coef_[0][3:]) / 2

# Evaluate: can we predict preferences on new comparisons?
X_test, y_test, _ = bt_model.generate_synthetic_preferences(n_pairs=50)
accuracy = trained_model.score(X_test, y_test)

# Visualize learned vs true reward function
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Compare learned vs true weights
features = ['Informativeness', 'Conciseness', 'Accuracy']
x_pos = np.arange(len(features))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, true_weights, width, label='True Weights',
                color='#2E7D32', alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, learned_weights, width, label='Learned Weights',
                color='#1976D2', alpha=0.8)

ax1.set_ylabel('Weight Value', fontsize=11)
ax1.set_title('Reward Function Recovery\n(Learning Human Preferences)',
              fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(features, fontsize=10)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# Right: Predicted vs actual preferences on test set
y_pred_proba = trained_model.predict_proba(X_test)[:, 1]
correct = (trained_model.predict(X_test) == y_test)

ax2.scatter(np.arange(len(y_test)), y_pred_proba,
           c=['#2E7D32' if c else '#C62828' for c in correct],
           alpha=0.6, s=80)
ax2.axhline(0.5, color='black', linestyle='--', alpha=0.5, linewidth=2)
ax2.set_xlabel('Test Example', fontsize=11)
ax2.set_ylabel('P(Response A Preferred)', fontsize=11)
ax2.set_title('Preference Predictions on Test Set\n(Green=Correct, Red=Incorrect)',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-0.05, 1.05])

# Add legend
legend_elements = [Patch(facecolor='#2E7D32', label=f'Correct ({correct.sum()}/{len(y_test)})'),
                   Patch(facecolor='#C62828', label=f'Incorrect ({(~correct).sum()}/{len(y_test)})')]
ax2.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-22/ch65/diagrams/preference_learning.png', dpi=150, bbox_inches='tight')
print("Generated: preference_learning.png")
print(f"Learned weights: Informativeness={learned_weights[0]:.2f}, Conciseness={learned_weights[1]:.2f}, Accuracy={learned_weights[2]:.2f}")
print(f"Test accuracy: {accuracy:.3f}")
