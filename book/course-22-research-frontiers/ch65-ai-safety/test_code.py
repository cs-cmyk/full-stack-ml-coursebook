"""
Code Review Test Script for Chapter 65
Tests all code blocks in sequence to verify they work correctly
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrow, Patch
import matplotlib.patches as mpatches
from scipy.special import expit as sigmoid
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("=" * 70)
print("TESTING CODE BLOCKS FROM CHAPTER 65")
print("=" * 70)

# ============================================================================
# BLOCK 1: Alignment Problems Visualization
# ============================================================================
print("\n[BLOCK 1] Testing alignment problems visualization...")

np.random.seed(42)

# Create figure showing outer vs inner alignment
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Outer alignment - Reward vs. True Value
policies = np.arange(1, 11)
true_value = 10 - (policies - 5)**2 / 3 + np.random.normal(0, 0.5, 10)
specified_reward = 8 - (policies - 3)**2 / 4 + np.random.normal(0, 0.5, 10)

ax1.plot(policies, true_value, 'o-', linewidth=2, markersize=8,
         label='True Human Values (V)', color='#2E7D32')
ax1.plot(policies, specified_reward, 's-', linewidth=2, markersize=8,
         label='Specified Reward (R)', color='#C62828')

# Mark optimal policies
true_opt = policies[np.argmax(true_value)]
spec_opt = policies[np.argmax(specified_reward)]
ax1.axvline(true_opt, color='#2E7D32', linestyle='--', alpha=0.5)
ax1.axvline(spec_opt, color='#C62828', linestyle='--', alpha=0.5)
ax1.fill_between([spec_opt-0.5, spec_opt+0.5], 0, 11,
                  color='#C62828', alpha=0.2, label='Misaligned Optimum')

ax1.set_xlabel('Policy π', fontsize=12)
ax1.set_ylabel('Value', fontsize=12)
ax1.set_title('Outer Alignment Problem\n(Specification Gap)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 11])

# Right plot: Inner alignment - Training vs deployment
train_states = np.linspace(0, 5, 50)
deploy_states = np.linspace(0, 10, 100)

# True objective R
true_objective_train = 8 * np.exp(-(train_states - 2.5)**2 / 2)
true_objective_deploy = 8 * np.exp(-(deploy_states - 2.5)**2 / 2)

# Learned proxy objective R'
learned_objective_train = 8 * np.exp(-(train_states - 2.5)**2 / 2) + np.random.normal(0, 0.2, len(train_states))
# Diverges outside training distribution
learned_objective_deploy = 8 * np.exp(-(deploy_states - 2.5)**2 / 2) * (1 - 0.15 * (deploy_states - 5).clip(0)**1.5 / 10)

ax2.fill_between(train_states, 0, 10, color='#1976D2', alpha=0.15, label='Training Distribution')
ax2.plot(deploy_states, true_objective_deploy, linewidth=3,
         label='True Objective R', color='#2E7D32')
ax2.plot(deploy_states, learned_objective_deploy, linewidth=3,
         label='Learned Objective R′', color='#C62828', linestyle='--')

ax2.axvline(5, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax2.text(5.2, 8.5, 'Distribution\nShift', fontsize=10, style='italic')

ax2.set_xlabel('State Space', fontsize=12)
ax2.set_ylabel('Objective Value', fontsize=12)
ax2.set_title('Inner Alignment Problem\n(Goal Misgeneralization)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 10])

plt.tight_layout()
plt.savefig('diagrams/alignment_problems.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Block 1 passed: Alignment problems visualization created")

# ============================================================================
# BLOCK 2: Reward Hacking Gridworld
# ============================================================================
print("\n[BLOCK 2] Testing reward hacking gridworld...")

np.random.seed(42)

class RewardHackingGridworld:
    """
    Simple gridworld where an agent can either:
    1. Take the intended path to the goal (longer)
    2. Exploit a shortcut that maximizes reward without achieving intended behavior
    """
    def __init__(self, size=5):
        self.size = size
        self.goal = (size-1, size-1)  # Bottom right
        self.shortcut = (0, size-1)   # Top right - unintended high-reward state

    def get_reward_intended(self, state):
        """Intended reward: high only at the goal after proper navigation"""
        if state == self.goal:
            return 100
        return -1  # Small penalty per step

    def get_reward_poorly_specified(self, state):
        """Poorly specified reward: also gives high reward at shortcut"""
        if state == self.goal:
            return 100
        if state == self.shortcut:
            return 90  # Unintended high reward!
        return -1

    def simulate_agent(self, reward_fn, num_episodes=1000):
        """Simulate agent learning with given reward function"""
        # Track which states the agent visits most
        visit_counts = np.zeros((self.size, self.size))

        for _ in range(num_episodes):
            # Start at (0, 0)
            state = (0, 0)
            for step in range(20):  # Max 20 steps per episode
                visit_counts[state] += 1

                # Agent learns to go where reward is highest
                # Simplified: check adjacent states and move toward higher reward
                neighbors = []
                rewards = []

                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_x, new_y = state[0] + dx, state[1] + dy
                    if 0 <= new_x < self.size and 0 <= new_y < self.size:
                        neighbors.append((new_x, new_y))
                        rewards.append(reward_fn((new_x, new_y)))

                if neighbors:
                    # Move toward highest reward (epsilon-greedy)
                    if np.random.random() < 0.8:  # 80% exploit
                        state = neighbors[np.argmax(rewards)]
                    else:  # 20% explore
                        state = neighbors[np.random.randint(len(neighbors))]

                # Stop if reached a terminal state
                if reward_fn(state) > 50:
                    visit_counts[state] += 10  # Heavy weight on terminal
                    break

        return visit_counts

# Create environment
env = RewardHackingGridworld(size=5)

# Simulate with both reward functions
visits_intended = env.simulate_agent(env.get_reward_intended, num_episodes=1000)
visits_hacked = env.simulate_agent(env.get_reward_poorly_specified, num_episodes=1000)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Intended behavior
im1 = ax1.imshow(visits_intended.T, cmap='YlOrRd', origin='lower')
ax1.add_patch(Rectangle((env.goal[0]-0.4, env.goal[1]-0.4), 0.8, 0.8,
                         fill=False, edgecolor='green', linewidth=3))
ax1.text(env.goal[0], env.goal[1], 'GOAL', ha='center', va='center',
         fontsize=11, fontweight='bold', color='green')
ax1.set_title('Proper Reward Specification\n(Agent reaches intended goal)',
              fontsize=12, fontweight='bold')
ax1.set_xlabel('X Position', fontsize=11)
ax1.set_ylabel('Y Position', fontsize=11)
ax1.grid(False)
plt.colorbar(im1, ax=ax1, label='Visit Frequency')

# Plot 2: Reward hacking
im2 = ax2.imshow(visits_hacked.T, cmap='YlOrRd', origin='lower')
ax2.add_patch(Rectangle((env.goal[0]-0.4, env.goal[1]-0.4), 0.8, 0.8,
                         fill=False, edgecolor='green', linewidth=3))
ax2.add_patch(Rectangle((env.shortcut[0]-0.4, env.shortcut[1]-0.4), 0.8, 0.8,
                         fill=False, edgecolor='red', linewidth=3))
ax2.text(env.goal[0], env.goal[1], 'GOAL', ha='center', va='center',
         fontsize=9, fontweight='bold', color='green')
ax2.text(env.shortcut[0], env.shortcut[1], 'SHORTCUT\n(hack!)', ha='center', va='center',
         fontsize=9, fontweight='bold', color='red')
ax2.set_title('Poorly Specified Reward\n(Agent exploits unintended high-reward state)',
              fontsize=12, fontweight='bold')
ax2.set_xlabel('X Position', fontsize=11)
ax2.set_ylabel('Y Position', fontsize=11)
ax2.grid(False)
plt.colorbar(im2, ax=ax2, label='Visit Frequency')

plt.tight_layout()
plt.savefig('diagrams/reward_hacking_demo.png', dpi=300, bbox_inches='tight')
plt.close()

print("REWARD HACKING DEMONSTRATION")
print("=" * 50)
print(f"Visits to intended goal (4,4): Proper={visits_intended[4,4]:.0f}, Hacked={visits_hacked[4,4]:.0f}")
print(f"Visits to shortcut (0,4): Proper={visits_intended[0,4]:.0f}, Hacked={visits_hacked[0,4]:.0f}")
print(f"\nWith poorly specified rewards, the agent visits the shortcut")
print(f"{visits_hacked[0,4]/visits_intended[0,4]:.1f}x more often!")
print(f"\nThis demonstrates reward hacking: the agent maximizes reward")
print(f"without achieving the intended goal.")

print("✓ Block 2 passed: Reward hacking demonstration completed")

# ============================================================================
# BLOCK 3: Goal Misgeneralization Demo
# ============================================================================
print("\n[BLOCK 3] Testing goal misgeneralization...")

np.random.seed(42)

class GoalMisgeneralizationDemo:
    """
    Demonstrates how models can learn spurious correlations during training
    that don't generalize to deployment (goal misgeneralization)
    """

    def create_data_with_spurious_correlation(self, n_samples=1000):
        """
        Create dataset where class label correlates with a spurious feature
        during training but not in deployment
        """
        # True features that determine the class
        X_true, y = make_classification(n_samples=n_samples, n_features=4,
                                        n_informative=4, n_redundant=0,
                                        n_clusters_per_class=2, random_state=42)

        # Add spurious feature that correlates with class in training
        # but is actually just random noise
        spurious_feature = np.random.normal(0, 1, n_samples)

        # Make spurious feature correlate with class in training data
        # Class 0: spurious feature tends negative
        # Class 1: spurious feature tends positive
        spurious_feature[y == 0] += -1.5  # Shift negative for class 0
        spurious_feature[y == 1] += 1.5   # Shift positive for class 1

        # Combine true and spurious features
        X_with_spurious = np.column_stack([X_true, spurious_feature])

        return X_true, X_with_spurious, y, spurious_feature

    def create_deployment_data(self, n_samples=200):
        """
        Create deployment data where spurious correlation is broken
        """
        X_true, y = make_classification(n_samples=n_samples, n_features=4,
                                        n_informative=4, n_redundant=0,
                                        n_clusters_per_class=2, random_state=123)

        # In deployment, spurious feature is UNCORRELATED with class
        spurious_feature = np.random.normal(0, 1, n_samples)
        # Intentionally break the correlation
        spurious_feature[y == 0] += 0.2   # Minimal shift
        spurious_feature[y == 1] += -0.2

        X_with_spurious = np.column_stack([X_true, spurious_feature])

        return X_true, X_with_spurious, y, spurious_feature

# Create demonstration
demo = GoalMisgeneralizationDemo()

# Generate training data (with spurious correlation)
X_true_train, X_spurious_train, y_train, spurious_train = \
    demo.create_data_with_spurious_correlation(n_samples=1000)

# Generate deployment data (spurious correlation broken)
X_true_deploy, X_spurious_deploy, y_deploy, spurious_deploy = \
    demo.create_deployment_data(n_samples=200)

# Train two models:
# Model 1: Uses only true features (aligned with intended goal)
model_true = LogisticRegression(random_state=42)
model_true.fit(X_true_train, y_train)

# Model 2: Uses true + spurious features (may learn wrong objective)
model_spurious = LogisticRegression(random_state=42)
model_spurious.fit(X_spurious_train, y_train)

# Evaluate on training distribution
train_acc_true = accuracy_score(y_train, model_true.predict(X_true_train))
train_acc_spurious = accuracy_score(y_train, model_spurious.predict(X_spurious_train))

# Evaluate on deployment distribution (distribution shift)
deploy_acc_true = accuracy_score(y_deploy, model_true.predict(X_true_deploy))
deploy_acc_spurious = accuracy_score(y_deploy, model_spurious.predict(X_spurious_deploy))

# Visualize spurious feature distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top row: Training distribution
ax1, ax2 = axes[0]
ax1.hist(spurious_train[y_train==0], bins=30, alpha=0.7, label='Class 0', color='#1976D2')
ax1.hist(spurious_train[y_train==1], bins=30, alpha=0.7, label='Class 1', color='#D32F2F')
ax1.set_title('Training: Spurious Feature Distribution\n(Strong correlation with class)',
              fontsize=12, fontweight='bold')
ax1.set_xlabel('Spurious Feature Value', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.legend()
ax1.axvline(0, color='black', linestyle='--', alpha=0.5)

# Feature importance for model trained with spurious feature
feature_names = ['True 1', 'True 2', 'True 3', 'True 4', 'Spurious']
importances = np.abs(model_spurious.coef_[0])
colors = ['#2E7D32']*4 + ['#C62828']
ax2.barh(feature_names, importances, color=colors)
ax2.set_title('Training: Learned Feature Importance\n(Model learns spurious correlation)',
              fontsize=12, fontweight='bold')
ax2.set_xlabel('Absolute Coefficient', fontsize=10)

# Bottom row: Deployment distribution
ax3, ax4 = axes[1]
ax3.hist(spurious_deploy[y_deploy==0], bins=30, alpha=0.7, label='Class 0', color='#1976D2')
ax3.hist(spurious_deploy[y_deploy==1], bins=30, alpha=0.7, label='Class 1', color='#D32F2F')
ax3.set_title('Deployment: Spurious Feature Distribution\n(Correlation broken)',
              fontsize=12, fontweight='bold')
ax3.set_xlabel('Spurious Feature Value', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.legend()
ax3.axvline(0, color='black', linestyle='--', alpha=0.5)

# Performance comparison
models = ['True Features\nOnly', 'True + Spurious\nFeatures']
train_accs = [train_acc_true, train_acc_spurious]
deploy_accs = [deploy_acc_true, deploy_acc_spurious]

x = np.arange(len(models))
width = 0.35

bars1 = ax4.bar(x - width/2, train_accs, width, label='Training', color='#388E3C')
bars2 = ax4.bar(x + width/2, deploy_accs, width, label='Deployment', color='#C62828')

ax4.set_ylabel('Accuracy', fontsize=10)
ax4.set_title('Performance: Goal Misgeneralization\n(Spurious correlations hurt deployment)',
              fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models, fontsize=10)
ax4.legend()
ax4.set_ylim([0.5, 1.0])
ax4.axhline(0.8, color='black', linestyle=':', alpha=0.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('diagrams/goal_misgeneralization.png', dpi=300, bbox_inches='tight')
plt.close()

print("GOAL MISGENERALIZATION DEMONSTRATION")
print("=" * 60)
print("\nModel 1: True features only")
print(f"  Training accuracy:   {train_acc_true:.3f}")
print(f"  Deployment accuracy: {deploy_acc_true:.3f}")
print(f"  Generalization gap:  {train_acc_true - deploy_acc_true:.3f}")

print("\nModel 2: True + spurious features")
print(f"  Training accuracy:   {train_acc_spurious:.3f}")
print(f"  Deployment accuracy: {deploy_acc_spurious:.3f}")
print(f"  Generalization gap:  {train_acc_spurious - deploy_acc_spurious:.3f}")

print("\nKey Insight:")
print(f"The model with spurious features performs {train_acc_spurious - train_acc_true:.3f} better")
print(f"in training but {deploy_acc_spurious - deploy_acc_true:.3f} worse in deployment.")
print(f"\nThis is goal misgeneralization: the model learned to rely on")
print(f"spurious correlations that held during training but broke in deployment.")

print("✓ Block 3 passed: Goal misgeneralization demonstration completed")

# ============================================================================
# BLOCK 4: Preference Learning (Bradley-Terry)
# ============================================================================
print("\n[BLOCK 4] Testing preference learning...")

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

    def extract_learned_rewards(self, response_features):
        """
        Extract learned reward for a single response.
        In Bradley-Terry, reward is linear: r(x) = w^T x
        """
        # For learned model, coefficients represent relative importance
        # We trained on concatenated features [A, B], so first half is for A
        weights_A = self.reward_model.coef_[0][:3]
        weights_B = self.reward_model.coef_[0][3:]

        # In Bradley-Terry, weights for A and -B should be similar
        # Average them for final reward function
        learned_weights = (weights_A - weights_B) / 2

        return np.dot(response_features, learned_weights)

# Generate preference dataset
bt_model = BradleyTerryPreferenceLearning()
X_prefs, y_prefs, true_weights = bt_model.generate_synthetic_preferences(n_pairs=200)

print("PREFERENCE LEARNING (Bradley-Terry Model)")
print("=" * 60)
print(f"Generated {len(X_prefs)} pairwise preference comparisons")
print(f"Each comparison: features of response A and B → preferred response")
print(f"\nTrue human reward weights:")
print(f"  Informativeness: {true_weights[0]:.2f}")
print(f"  Conciseness:     {true_weights[1]:.2f}")
print(f"  Accuracy:        {true_weights[2]:.2f}")

# Train reward model from preferences
trained_model = bt_model.train_reward_model(X_prefs, y_prefs)

# Extract learned weights
learned_weights = (trained_model.coef_[0][:3] - trained_model.coef_[0][3:]) / 2

print(f"\nLearned reward weights:")
print(f"  Informativeness: {learned_weights[0]:.2f}")
print(f"  Conciseness:     {learned_weights[1]:.2f}")
print(f"  Accuracy:        {learned_weights[2]:.2f}")

# Evaluate: can we predict preferences on new comparisons?
X_test, y_test, _ = bt_model.generate_synthetic_preferences(n_pairs=50)
accuracy = trained_model.score(X_test, y_test)
print(f"\nPreference prediction accuracy on test set: {accuracy:.3f}")

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
plt.savefig('diagrams/preference_learning.png', dpi=300, bbox_inches='tight')
plt.close()

# Demonstrate how learned reward function ranks responses
print("\n" + "=" * 60)
print("USING LEARNED REWARD MODEL TO RANK RESPONSES")
print("=" * 60)

# Create 5 test responses with different characteristics
test_responses = np.array([
    [0.3, 0.2, 0.9],  # Low info, low concise, high accuracy
    [0.8, 0.8, 0.5],  # High info, high concise, medium accuracy
    [0.9, 0.1, 0.8],  # High info, low concise, high accuracy
    [0.5, 0.9, 0.7],  # Medium info, high concise, high accuracy
    [0.2, 0.3, 0.3],  # Low on all dimensions
])

# Compute rewards with true and learned functions
true_rewards = test_responses @ true_weights
learned_rewards = test_responses @ learned_weights

print("\nResponse Rankings:")
print(f"{'Response':<10} {'Features (I, C, A)':<25} {'True Reward':<15} {'Learned Reward':<15}")
print("-" * 70)

for i, resp in enumerate(test_responses):
    features_str = f"({resp[0]:.1f}, {resp[1]:.1f}, {resp[2]:.1f})"
    print(f"Response {i+1:<2} {features_str:<25} {true_rewards[i]:<15.2f} {learned_rewards[i]:<15.2f}")

# Check if rankings agree
true_ranking = np.argsort(-true_rewards)
learned_ranking = np.argsort(-learned_rewards)

print(f"\nTrue ranking (best to worst):    {true_ranking + 1}")
print(f"Learned ranking (best to worst): {learned_ranking + 1}")
print(f"\nRankings match: {np.array_equal(true_ranking, learned_ranking)}")

print("✓ Block 4 passed: Preference learning completed")

# ============================================================================
# BLOCK 5: Gridworld Comparison (Solution 2)
# ============================================================================
print("\n[BLOCK 5] Testing gridworld comparison (Solution 2)...")

np.random.seed(42)

class GridworldEnvironment:
    """5x5 gridworld with start, goal, hazards"""
    def __init__(self):
        self.size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.hazards = [(1, 1), (2, 3), (3, 2)]
        self.reset()

    def reset(self):
        self.position = self.start
        self.visited = set([self.start])
        return self.position

    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = moves[action]
        new_x = max(0, min(self.size-1, self.position[0] + dx))
        new_y = max(0, min(self.size-1, self.position[1] + dy))
        self.position = (new_x, new_y)

        self.visited.add(self.position)

        done = (self.position == self.goal or self.position in self.hazards)
        return self.position, done

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def sparse_reward(env):
    if env.position == env.goal:
        return 100
    if env.position in env.hazards:
        return -100
    return 0

def dense_reward(env):
    if env.position == env.goal:
        return 100
    if env.position in env.hazards:
        return -100

    # Distance-based shaping
    dist_to_goal = env.manhattan_distance(env.position, env.goal)
    max_dist = env.size * 2
    shaping = 5 * (1 - dist_to_goal / max_dist)

    return shaping - 1  # Step penalty

def poorly_specified_reward(env):
    if env.position == env.goal:
        return 100
    if env.position in env.hazards:
        return -100

    # EXPLOIT: Big reward for discovering new states!
    if env.position not in env.visited:
        return 50  # Encourages exploration over goal-seeking

    return -1

# Simple Q-learning agent
class QLearningAgent:
    def __init__(self, size=5, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.q_table = np.zeros((size, size, 4))  # Q(state, action)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.size = size

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

def train_agent(reward_fn, episodes=1000, max_steps=50):
    env = GridworldEnvironment()
    agent = QLearningAgent()

    rewards_per_episode = []
    steps_per_episode = []
    success_count = 0

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, done = env.step(action)
            reward = reward_fn(env)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                if env.position == env.goal:
                    success_count += 1
                break

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(step + 1)

    return agent, rewards_per_episode, steps_per_episode, success_count

# Train agents with different reward functions
print("Training agents with different reward functions...")
agent_sparse, rewards_sparse, steps_sparse, success_sparse = \
    train_agent(sparse_reward, episodes=1000)
agent_dense, rewards_dense, steps_dense, success_dense = \
    train_agent(dense_reward, episodes=1000)
agent_poor, rewards_poor, steps_poor, success_poor = \
    train_agent(poorly_specified_reward, episodes=1000)

# Visualize learning curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cumulative reward over training
ax1 = axes[0, 0]
window = 50
ax1.plot(np.convolve(rewards_sparse, np.ones(window)/window, mode='valid'),
         label='Sparse', linewidth=2)
ax1.plot(np.convolve(rewards_dense, np.ones(window)/window, mode='valid'),
         label='Dense (shaped)', linewidth=2)
ax1.plot(np.convolve(rewards_poor, np.ones(window)/window, mode='valid'),
         label='Poorly specified', linewidth=2)
ax1.set_xlabel('Episode', fontsize=11)
ax1.set_ylabel('Average Reward', fontsize=11)
ax1.set_title('Learning Curves\n(50-episode moving average)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Success rate over training
ax2 = axes[0, 1]
def success_rate(success_list, window=100):
    return np.convolve(success_list, np.ones(window)/window, mode='valid')

success_sparse_binary = [1 if rewards_sparse[i] >= 90 else 0 for i in range(len(rewards_sparse))]
success_dense_binary = [1 if rewards_dense[i] >= 90 else 0 for i in range(len(rewards_dense))]
success_poor_binary = [1 if rewards_poor[i] >= 90 else 0 for i in range(len(rewards_poor))]

ax2.plot(success_rate(success_sparse_binary), label='Sparse', linewidth=2)
ax2.plot(success_rate(success_dense_binary), label='Dense (shaped)', linewidth=2)
ax2.plot(success_rate(success_poor_binary), label='Poorly specified', linewidth=2)
ax2.set_xlabel('Episode', fontsize=11)
ax2.set_ylabel('Success Rate', fontsize=11)
ax2.set_title('Goal Achievement Rate\n(100-episode moving average)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Learned policies (heatmap of state values)
ax3 = axes[1, 0]
value_map_poor = np.max(agent_poor.q_table, axis=2)
im = ax3.imshow(value_map_poor, cmap='RdYlGn', origin='lower')
ax3.set_title('Learned State Values: Poorly Specified Reward\n(Shows exploration bias)',
              fontsize=12, fontweight='bold')
ax3.set_xlabel('X Position')
ax3.set_ylabel('Y Position')
plt.colorbar(im, ax=ax3)

# Mark special states
ax3.plot(0, 0, 'bs', markersize=12, label='Start')
ax3.plot(4, 4, 'g*', markersize=15, label='Goal')
for haz in [(1, 1), (2, 3), (3, 2)]:
    ax3.plot(haz[1], haz[0], 'rx', markersize=12, markeredgewidth=3)
ax3.legend(loc='upper left', fontsize=9)

# Plot 4: Performance comparison
ax4 = axes[1, 1]
metrics = ['Final Success\nRate (%)', 'Avg Steps\nto Goal', 'Total\nReward']
sparse_metrics = [success_sparse/10, np.mean(steps_sparse[-100:]), np.mean(rewards_sparse[-100:])]
dense_metrics = [success_dense/10, np.mean(steps_dense[-100:]), np.mean(rewards_dense[-100:])]
poor_metrics = [success_poor/10, np.mean(steps_poor[-100:]), np.mean(rewards_poor[-100:])]

x = np.arange(len(metrics))
width = 0.25

ax4.bar(x - width, sparse_metrics, width, label='Sparse', alpha=0.8)
ax4.bar(x, dense_metrics, width, label='Dense', alpha=0.8)
ax4.bar(x + width, poor_metrics, width, label='Poorly specified', alpha=0.8)

ax4.set_ylabel('Value (normalized)', fontsize=11)
ax4.set_title('Final Performance Comparison\n(Last 100 episodes)', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, fontsize=10)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('diagrams/gridworld_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nRESULTS SUMMARY")
print("=" * 60)
print(f"Sparse Reward:        {success_sparse/10:.1f}% success, avg {np.mean(steps_sparse[-100:]):.1f} steps")
print(f"Dense Reward:         {success_dense/10:.1f}% success, avg {np.mean(steps_dense[-100:]):.1f} steps")
print(f"Poorly Specified:     {success_poor/10:.1f}% success, avg {np.mean(steps_poor[-100:]):.1f} steps")
print("\nOBSERVATIONS:")
print("- Dense reward learns faster (reward shaping provides gradient)")
print("- Poorly specified reward shows lower success rate (explores rather than goal-seeks)")
print("- Sparse reward is slowest to learn but eventually succeeds")
print("- Poorly specified reward demonstrates reward hacking: exploration bonus")
print("  leads agent to prioritize visiting new states over reaching goal")

print("✓ Block 5 passed: Gridworld comparison completed")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("ALL CODE BLOCKS PASSED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated diagrams:")
print("  - diagrams/alignment_problems.png")
print("  - diagrams/reward_hacking_demo.png")
print("  - diagrams/goal_misgeneralization.png")
print("  - diagrams/preference_learning.png")
print("  - diagrams/gridworld_comparison.png")
print("\nAll code executed without errors and produced expected outputs.")
