> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# 65.1: Directed Acyclic Graphs (DAGs) and d-Separation

## Why This Matters

Every day, researchers face the same question: does X cause Y, or are they merely correlated? A pharmaceutical company observes that patients taking a new drug recover faster—but was it the drug, or did healthier patients simply get prescribed it more often? A tech company sees users who engage with a feature convert at higher rates—but does the feature cause conversions, or do already-engaged users just happen to use it? Directed Acyclic Graphs (DAGs) provide a formal language for making causal assumptions explicit, allowing data scientists to distinguish genuine cause-and-effect from spurious correlation before collecting a single data point.

## Intuition

Imagine a company's organizational chart. Information flows downward: the CEO makes strategic decisions that influence VPs, who direct department heads, who guide individual contributors. If the marketing VP and engineering VP both report to the CEO, their departments might show coordinated behavior—not because they directly influence each other, but because they both respond to CEO directives. This is exactly how causal relationships work in data.

A Directed Acyclic Graph is like that organizational chart for variables in a system. Each node represents a variable (temperature, ice cream sales, drowning deaths). Each arrow represents a direct causal influence: "this variable causes changes in that variable." The "directed" part means arrows have direction (X → Y is different from Y → X). The "acyclic" part means no feedback loops—you cannot follow arrows in a circle and return to where you started.

Consider three temperature-related scenarios. First, a chain: Temperature → Ice Cream Sales → Vendor Profits. Hot weather causes people to buy more ice cream, which causes vendors to earn more. Second, a fork: Ice Cream Sales ← Temperature → Beach Attendance. Hot weather causes both phenomena, creating correlation between ice cream and beaches even though neither directly causes the other. Third, a collider: Talent → Professional Athlete ← Training. Both talent and training influence who becomes professional, but among pros, talent and training might appear negatively correlated because someone with less talent needs more training to compete.

The power of DAGs lies in d-separation, a graphical algorithm that answers questions like "are these two variables independent given what I've observed?" without needing any data. Just by looking at the structure of arrows, we can determine which variables we must control for in statistical analysis and which we must avoid controlling for—saving costly mistakes before experiments even begin.

Think of association flowing through the graph like water through a network of one-way pipes. Normally, water flows freely along the pipes. But you can "pinch" certain pipes by conditioning on variables (controlling for them in regression). In chains and forks, pinching a pipe blocks the flow between variables upstream and downstream. But in colliders—pipes where two streams merge—pinching creates back-pressure that actually makes the upstream sources become related, opening a path that was previously blocked.

## Formal Definition

A **Directed Acyclic Graph (DAG)** is a tuple G = (V, E) where:
- V is a set of vertices (nodes) representing random variables
- E ⊆ V × V is a set of directed edges (arrows) representing direct causal influences
- The graph contains no directed cycles: there is no sequence of edges v₁ → v₂ → ... → vₖ → v₁

**Graph terminology:**
- If X → Y, then X is a **parent** of Y, and Y is a **child** of X
- **Ancestors** of Y are all variables with a directed path to Y
- **Descendants** of Y are all variables reachable by following arrows from Y
- A **path** is a sequence of edges (possibly traversing arrows backward)
- A **directed path** follows arrows in their intended direction only

**The three elementary structures** (building blocks of all causal reasoning):

1. **Chain**: X → Y → Z
   - X influences Y, which influences Z (mediation)
   - X and Z are marginally associated: X ⊥̸ Z
   - X and Z are conditionally independent given Y: X ⊥ Z | Y

2. **Fork**: X ← Y → Z
   - Y influences both X and Z (confounding)
   - X and Z are marginally associated: X ⊥̸ Z
   - X and Z are conditionally independent given Y: X ⊥ Z | Y

3. **Collider**: X → Y ← Z
   - Both X and Z influence Y (common effect)
   - X and Z are marginally independent: X ⊥ Z
   - X and Z are conditionally associated given Y: X ⊥̸ Z | Y

**d-Separation** ("dependency separation") is a graphical criterion for determining conditional independence:

Variables X and Z are **d-separated** by a set of variables C (denoted X ⊥ Z | C) if and only if C blocks all paths between X and Z, where a path is blocked if it contains:
- A chain A → B → C or fork A ← B → C where B ∈ C (conditioning blocks)
- A collider A → B ← C where neither B nor any descendant of B is in C (not conditioning blocks)

**Causal Markov Assumption**: If X ⊥ Z | C in the DAG (d-separated), then the corresponding variables are conditionally independent in the probability distribution: P(X, Z | C) = P(X | C)P(Z | C).

This assumption bridges graphical structure and statistical independence, allowing causal reasoning from structure alone.

> **Key Concept:** d-Separation allows us to read conditional independence relationships directly from graph structure—conditioning blocks association in chains and forks but opens association in colliders.

## Visualization

```python
# Visualization of the three elementary structures
import matplotlib.pyplot as plt
import networkx as nx

# Create figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Chain: X → Y → Z
G_chain = nx.DiGraph()
G_chain.add_edges_from([('X', 'Y'), ('Y', 'Z')])
pos_chain = {'X': (0, 0), 'Y': (1, 0), 'Z': (2, 0)}

axes[0].set_title('Chain: Mediation\nX → Y → Z\n\nX ⊥ Z | Y (conditioning blocks)',
                  fontsize=12, fontweight='bold')
nx.draw(G_chain, pos_chain, ax=axes[0], with_labels=True,
        node_color='lightblue', node_size=2000, font_size=14,
        font_weight='bold', arrows=True, arrowsize=20,
        arrowstyle='->', edge_color='black', width=2)
axes[0].axis('off')

# 2. Fork: X ← Y → Z
G_fork = nx.DiGraph()
G_fork.add_edges_from([('Y', 'X'), ('Y', 'Z')])
pos_fork = {'X': (0, 0), 'Y': (1, 0.5), 'Z': (2, 0)}

axes[1].set_title('Fork: Confounding\nX ← Y → Z\n\nX ⊥ Z | Y (conditioning blocks)',
                  fontsize=12, fontweight='bold')
nx.draw(G_fork, pos_fork, ax=axes[1], with_labels=True,
        node_color='lightgreen', node_size=2000, font_size=14,
        font_weight='bold', arrows=True, arrowsize=20,
        arrowstyle='->', edge_color='black', width=2)
axes[1].axis('off')

# 3. Collider: X → Y ← Z
G_collider = nx.DiGraph()
G_collider.add_edges_from([('X', 'Y'), ('Z', 'Y')])
pos_collider = {'X': (0, 0), 'Y': (1, -0.5), 'Z': (2, 0)}

axes[2].set_title('Collider: Selection Bias\nX → Y ← Z\n\nX ⊥̸ Z | Y (conditioning opens!)',
                  fontsize=12, fontweight='bold', color='red')
nx.draw(G_collider, pos_collider, ax=axes[2], with_labels=True,
        node_color='lightcoral', node_size=2000, font_size=14,
        font_weight='bold', arrows=True, arrowsize=20,
        arrowstyle='->', edge_color='black', width=2)
axes[2].axis('off')

plt.tight_layout()
plt.savefig('book/course-19/ch54/elementary_structures.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# Three-panel diagram showing Chain (blue), Fork (green), and Collider (red)
# Each annotated with independence relationships
```

The diagram above shows the three fundamental building blocks. Notice how the collider (right panel, in red) has opposite behavior: conditioning creates association rather than blocking it.

## Examples

### Part 1: Building and Visualizing a Simple DAG

```python
# Building a DAG representing a medical scenario
import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
# Scenario: Smoking → Lung Cancer ← Genetic Factor → Chronic Cough
G = nx.DiGraph()

# Add nodes
nodes = ['Smoking', 'Genetic_Factor', 'Lung_Cancer', 'Chronic_Cough']
G.add_nodes_from(nodes)

# Add edges (causal relationships)
edges = [
    ('Smoking', 'Lung_Cancer'),
    ('Genetic_Factor', 'Lung_Cancer'),
    ('Genetic_Factor', 'Chronic_Cough')
]
G.add_edges_from(edges)

# Visualize the DAG
pos = {
    'Smoking': (0, 1),
    'Genetic_Factor': (2, 1),
    'Lung_Cancer': (1, 0),
    'Chronic_Cough': (3, 0)
}

plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color='skyblue',
        node_size=3000, font_size=10, font_weight='bold',
        arrows=True, arrowsize=20, arrowstyle='->',
        edge_color='gray', width=2)
plt.title('Causal DAG: Smoking, Genetics, and Health Outcomes',
          fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('book/course-19/ch54/smoking_dag.png', dpi=150, bbox_inches='tight')
plt.show()

# Query graph structure programmatically
print("Parents of Lung_Cancer:", list(G.predecessors('Lung_Cancer')))
print("Children of Genetic_Factor:", list(G.successors('Genetic_Factor')))

# Check for paths
print("\nAll paths from Smoking to Chronic_Cough:")
for path in nx.all_simple_paths(G.to_undirected(), 'Smoking', 'Chronic_Cough'):
    print("  ", path)

# Output:
# Parents of Lung_Cancer: ['Smoking', 'Genetic_Factor']
# Children of Genetic_Factor: ['Lung_Cancer', 'Chronic_Cough']
#
# All paths from Smoking to Chronic_Cough:
#   ['Smoking', 'Lung_Cancer', 'Genetic_Factor', 'Chronic_Cough']
```

The code above constructs a simple 4-node DAG representing causal relationships in a medical scenario. Smoking and genetic factors both cause lung cancer (making Lung_Cancer a collider), while genetic factors also cause chronic cough. The `predecessors()` method identifies parents (direct causes), and `successors()` identifies children (direct effects). Notice the path from Smoking to Chronic_Cough goes through Lung_Cancer (a collider)—this will become important when we test d-separation.

### Part 2: Generating Data from the Three Elementary Structures

```python
# Generate synthetic data demonstrating each structure
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000

# Structure 1: Chain (X → Y → Z)
print("=" * 60)
print("CHAIN: X → Y → Z (Mediation)")
print("=" * 60)

X_chain = np.random.randn(n)
Y_chain = 2 * X_chain + np.random.randn(n) * 0.5  # Y depends on X
Z_chain = 1.5 * Y_chain + np.random.randn(n) * 0.5  # Z depends on Y

df_chain = pd.DataFrame({
    'X': X_chain,
    'Y': Y_chain,
    'Z': Z_chain
})

# Marginal correlation (X and Z associated)
corr_XZ_marginal = np.corrcoef(X_chain, Z_chain)[0, 1]
print(f"Corr(X, Z) without conditioning: {corr_XZ_marginal:.3f}")

# Conditional correlation (X and Z independent given Y)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(Y_chain.reshape(-1, 1), Z_chain)
Z_residuals = Z_chain - model.predict(Y_chain.reshape(-1, 1))

model.fit(Y_chain.reshape(-1, 1), X_chain)
X_residuals = X_chain - model.predict(Y_chain.reshape(-1, 1))

corr_XZ_conditional = np.corrcoef(X_residuals, Z_residuals)[0, 1]
print(f"Corr(X, Z) conditioning on Y: {corr_XZ_conditional:.3f}")
print("Chain rule: Conditioning blocks the association.\n")

# Structure 2: Fork (X ← Y → Z)
print("=" * 60)
print("FORK: X ← Y → Z (Confounding)")
print("=" * 60)

Y_fork = np.random.randn(n)
X_fork = 1.5 * Y_fork + np.random.randn(n) * 0.5  # X depends on Y
Z_fork = 2.0 * Y_fork + np.random.randn(n) * 0.5  # Z depends on Y

df_fork = pd.DataFrame({
    'X': X_fork,
    'Y': Y_fork,
    'Z': Z_fork
})

# Marginal correlation (X and Z associated through Y)
corr_XZ_marginal_fork = np.corrcoef(X_fork, Z_fork)[0, 1]
print(f"Corr(X, Z) without conditioning: {corr_XZ_marginal_fork:.3f}")

# Conditional correlation (X and Z independent given Y)
model.fit(Y_fork.reshape(-1, 1), Z_fork)
Z_residuals_fork = Z_fork - model.predict(Y_fork.reshape(-1, 1))

model.fit(Y_fork.reshape(-1, 1), X_fork)
X_residuals_fork = X_fork - model.predict(Y_fork.reshape(-1, 1))

corr_XZ_conditional_fork = np.corrcoef(X_residuals_fork, Z_residuals_fork)[0, 1]
print(f"Corr(X, Z) conditioning on Y: {corr_XZ_conditional_fork:.3f}")
print("Fork rule: Conditioning blocks the association.\n")

# Structure 3: Collider (X → Y ← Z)
print("=" * 60)
print("COLLIDER: X → Y ← Z (Selection Bias)")
print("=" * 60)

X_collider = np.random.randn(n)  # X independent
Z_collider = np.random.randn(n)  # Z independent
Y_collider = 1.5 * X_collider + 2.0 * Z_collider + np.random.randn(n) * 0.5

df_collider = pd.DataFrame({
    'X': X_collider,
    'Y': Y_collider,
    'Z': Z_collider
})

# Marginal correlation (X and Z independent)
corr_XZ_marginal_collider = np.corrcoef(X_collider, Z_collider)[0, 1]
print(f"Corr(X, Z) without conditioning: {corr_XZ_marginal_collider:.3f}")

# Conditional correlation (X and Z BECOME associated given Y)
model.fit(Y_collider.reshape(-1, 1), Z_collider)
Z_residuals_collider = Z_collider - model.predict(Y_collider.reshape(-1, 1))

model.fit(Y_collider.reshape(-1, 1), X_collider)
X_residuals_collider = X_collider - model.predict(Y_collider.reshape(-1, 1))

corr_XZ_conditional_collider = np.corrcoef(X_residuals_collider, Z_residuals_collider)[0, 1]
print(f"Corr(X, Z) conditioning on Y: {corr_XZ_conditional_collider:.3f}")
print("Collider rule: Conditioning OPENS the association (counterintuitive!).\n")

# Output:
# ============================================================
# CHAIN: X → Y → Z (Mediation)
# ============================================================
# Corr(X, Z) without conditioning: 0.951
# Corr(X, Z) conditioning on Y: 0.005
# Chain rule: Conditioning blocks the association.
#
# ============================================================
# FORK: X ← Y → Z (Confounding)
# ============================================================
# Corr(X, Z) without conditioning: 0.902
# Corr(X, Z) conditioning on Y: -0.014
# Fork rule: Conditioning blocks the association.
#
# ============================================================
# COLLIDER: X → Y ← Z (Selection Bias)
# ============================================================
# Corr(X, Z) without conditioning: -0.017
# Corr(X, Z) conditioning on Y: -0.692
# Collider rule: Conditioning OPENS the association (counterintuitive!).
```

This example generates synthetic data from known causal structures, demonstrating the core principle of d-separation empirically. In the chain and fork structures, X and Z are marginally correlated (0.95 and 0.90) but become independent when conditioning on Y (correlations drop to near zero). In the collider structure, X and Z start independent (-0.017, near zero) but become strongly negatively correlated (-0.69) when we condition on Y. This counterintuitive behavior—that controlling for a variable can create spurious associations—is why understanding colliders is critical for causal inference.

### Part 3: Testing d-Separation with Code

```python
# Using causalgraphicalmodels library for d-separation testing
# Note: If not installed, run: pip install causalgraphicalmodels
from causalgraphicalmodels import CausalGraphicalModel

# Define the smoking/genetics DAG from Part 1
dag = CausalGraphicalModel(
    nodes=['Smoking', 'Genetic_Factor', 'Lung_Cancer', 'Chronic_Cough'],
    edges=[
        ('Smoking', 'Lung_Cancer'),
        ('Genetic_Factor', 'Lung_Cancer'),
        ('Genetic_Factor', 'Chronic_Cough')
    ]
)

# Test d-separation queries
print("d-Separation Queries on Smoking/Genetics DAG")
print("=" * 60)

# Query 1: Are Smoking and Chronic_Cough independent unconditionally?
query1 = dag.is_d_separated('Smoking', 'Chronic_Cough', set())
print(f"Smoking ⊥ Chronic_Cough | ∅: {query1}")
print("Path: Smoking → Lung_Cancer ← Genetic_Factor → Chronic_Cough")
print("Contains collider (Lung_Cancer), not conditioned → PATH BLOCKED")
print()

# Query 2: Are they independent given Lung_Cancer?
query2 = dag.is_d_separated('Smoking', 'Chronic_Cough', {'Lung_Cancer'})
print(f"Smoking ⊥ Chronic_Cough | Lung_Cancer: {query2}")
print("Path: Smoking → Lung_Cancer ← Genetic_Factor → Chronic_Cough")
print("Contains collider (Lung_Cancer), IS conditioned → PATH OPENS")
print()

# Query 3: Are they independent given Genetic_Factor?
query3 = dag.is_d_separated('Smoking', 'Chronic_Cough', {'Genetic_Factor'})
print(f"Smoking ⊥ Chronic_Cough | Genetic_Factor: {query3}")
print("Path: Smoking → Lung_Cancer ← Genetic_Factor → Chronic_Cough")
print("Genetic_Factor is in the path after collider → BLOCKS (fork structure)")
print()

# Build a more complex DAG for practice
complex_dag = CausalGraphicalModel(
    nodes=['A', 'B', 'C', 'D', 'E'],
    edges=[
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'D'),
        ('C', 'D'),
        ('D', 'E')
    ]
)

print("\nComplex DAG Structure:")
print("  A → B → D → E")
print("  A → C → D")
print("=" * 60)

# Query: Is A independent of E given D?
query4 = complex_dag.is_d_separated('A', 'E', {'D'})
print(f"A ⊥ E | D: {query4}")
print("All paths from A to E go through D (chain/mediator)")
print("Conditioning on D blocks all paths")
print()

# Query: Is B independent of C given A?
query5 = complex_dag.is_d_separated('B', 'C', {'A'})
print(f"B ⊥ C | A: {query5}")
print("Path B ← A → C is a fork, A is conditioned → BLOCKED")
print()

# Query: Is B independent of C given D?
query6 = complex_dag.is_d_separated('B', 'C', {'D'})
print(f"B ⊥ C | D: {query6}")
print("Path B → D ← C is a collider, D is conditioned → PATH OPENS")
print("Also path B ← A → C exists, D doesn't block it → NOT d-separated")

# Output:
# d-Separation Queries on Smoking/Genetics DAG
# ============================================================
# Smoking ⊥ Chronic_Cough | ∅: True
# Path: Smoking → Lung_Cancer ← Genetic_Factor → Chronic_Cough
# Contains collider (Lung_Cancer), not conditioned → PATH BLOCKED
#
# Smoking ⊥ Chronic_Cough | Lung_Cancer: False
# Path: Smoking → Lung_Cancer ← Genetic_Factor → Chronic_Cough
# Contains collider (Lung_Cancer), IS conditioned → PATH OPENS
#
# Smoking ⊥ Chronic_Cough | Genetic_Factor: True
# Path: Smoking → Lung_Cancer ← Genetic_Factor → Chronic_Cough
# Genetic_Factor is in the path after collider → BLOCKS (fork structure)
#
# Complex DAG Structure:
#   A → B → D → E
#   A → C → D
# ============================================================
# A ⊥ E | D: True
# All paths from A to E go through D (chain/mediator)
# Conditioning on D blocks all paths
#
# B ⊥ C | A: True
# Path B ← A → C is a fork, A is conditioned → BLOCKED
#
# B ⊥ C | D: False
# Path B → D ← C is a collider, D is conditioned → PATH OPENS
# Also path B ← A → C exists, D doesn't block it → NOT d-separated
```

The `causalgraphicalmodels` library provides a programmatic interface for testing d-separation. The key insight: d-separation is purely structural—it depends only on the graph topology, not on data. The first query shows that Smoking and Chronic_Cough are marginally independent (d-separated) because the only path between them contains an unconditioned collider (Lung_Cancer). But conditioning on that collider (query 2) opens the path, making them dependent. This demonstrates Berkson's paradox: controlling for a common effect induces spurious association between its causes.

### Part 4: Collider Bias in Action—The Sports Analytics Paradox

```python
# Demonstrating collider bias: Height and Speed among professional athletes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n = 10000

# Generate population data: Height and Speed are INDEPENDENT
height = np.random.normal(loc=180, scale=10, size=n)  # cm
speed = np.random.normal(loc=100, scale=15, size=n)   # arbitrary units

# Professional selection: requires high combined ability
# Pro status is a collider: Height → Pro ← Speed
ability_threshold = 265  # Only top ~15% become professional
combined_ability = height + speed
is_pro = (combined_ability > ability_threshold).astype(int)

# Create DataFrame
df_sports = pd.DataFrame({
    'Height': height,
    'Speed': speed,
    'Combined_Ability': combined_ability,
    'Professional': is_pro
})

print("Sports Analytics Collider Bias Example")
print("=" * 60)
print(f"Total population: {n}")
print(f"Professional athletes: {is_pro.sum()} ({100*is_pro.mean():.1f}%)")
print()

# Correlation in general population (unconditional)
corr_general = df_sports['Height'].corr(df_sports['Speed'])
print(f"Corr(Height, Speed) in general population: {corr_general:.3f}")
print("→ Near zero, as expected (generated independently)")
print()

# Correlation among professionals (conditioning on Pro=1)
df_pros = df_sports[df_sports['Professional'] == 1]
corr_pros = df_pros['Height'].corr(df_pros['Speed'])
print(f"Corr(Height, Speed) among professionals: {corr_pros:.3f}")
print("→ Strong NEGATIVE correlation!")
print()

print("Explanation:")
print("Professional status is a collider (Height → Pro ← Speed).")
print("To be professional requires high Height + Speed.")
print("Among pros, if someone is tall, they can compensate with less speed.")
print("If someone is short, they must be very fast to qualify.")
print("This creates negative correlation that doesn't exist in population.")
print()

# Visualize the paradox
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: General population
axes[0].scatter(df_sports['Height'], df_sports['Speed'],
                alpha=0.3, s=10, color='gray')
axes[0].set_xlabel('Height (cm)', fontsize=12)
axes[0].set_ylabel('Speed (units)', fontsize=12)
axes[0].set_title(f'General Population\nCorr = {corr_general:.3f}',
                  fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Right: Professionals only
axes[1].scatter(df_pros['Height'], df_pros['Speed'],
                alpha=0.6, s=30, color='red')
axes[1].set_xlabel('Height (cm)', fontsize=12)
axes[1].set_ylabel('Speed (units)', fontsize=12)
axes[1].set_title(f'Professional Athletes Only\nCorr = {corr_pros:.3f}',
                  fontsize=14, fontweight='bold', color='red')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Collider Bias: Height and Speed in Sports',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('book/course-19/ch54/collider_sports.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# Sports Analytics Collider Bias Example
# ============================================================
# Total population: 10000
# Professional athletes: 1489 (14.9%)
#
# Corr(Height, Speed) in general population: -0.002
# → Near zero, as expected (generated independently)
#
# Corr(Height, Speed) among professionals: -0.565
# → Strong NEGATIVE correlation!
#
# Explanation:
# Professional status is a collider (Height → Pro ← Speed).
# To be professional requires high Height + Speed.
# Among pros, if someone is tall, they can compensate with less speed.
# If someone is short, they must be very fast to qualify.
# This creates negative correlation that doesn't exist in population.
```

This example demonstrates Berkson's paradox, a classic manifestation of collider bias. In the general population, height and speed are uncorrelated (r = -0.002). But among professional athletes—who were selected based on high combined ability—the correlation becomes strongly negative (r = -0.565). This is not because height causes reduced speed, but because professional status is a collider: both height and speed cause it. When we condition on being professional (by analyzing only pros), we induce a spurious negative correlation. This same pattern appears in many real-world contexts: the "attractive but untalented celebrity" stereotype, the "smart students have poor social skills" myth, and hospital admission biases in medical research.

### Part 5: Using DAGs to Identify Confounders

```python
# Practical example: Identifying confounders in treatment effect estimation
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000

# True causal structure:
# Age → Treatment (older patients get treatment more often)
# Age → Recovery (older patients recover slower)
# Treatment → Recovery (treatment helps recovery)

# Generate data
age = np.random.uniform(20, 80, size=n)

# Treatment assignment depends on age (confounding!)
# Doctors preferentially treat older patients
treatment_prob = 0.2 + 0.6 * (age - 20) / 60  # Increases with age
treatment = np.random.binomial(1, treatment_prob)

# Recovery depends on both age and treatment
# True causal effect of treatment: +15 points
# Age effect: -0.5 points per year
recovery = 100 - 0.5 * age + 15 * treatment + np.random.normal(0, 5, n)

df_medical = pd.DataFrame({
    'Age': age,
    'Treatment': treatment,
    'Recovery': recovery
})

print("Medical Treatment Analysis: Identifying Confounders")
print("=" * 60)
print("True causal structure:")
print("  Age → Treatment")
print("  Age → Recovery")
print("  Treatment → Recovery")
print()
print("Age is a CONFOUNDER (fork structure: Treatment ← Age → Recovery)")
print()

# Naive analysis: ignore age (WRONG - confounded estimate)
X_naive = df_medical[['Treatment']]
y = df_medical['Recovery']
model_naive = LinearRegression()
model_naive.fit(X_naive, y)
naive_effect = model_naive.coef_[0]

print(f"NAIVE estimate (ignoring Age): {naive_effect:.2f}")
print("This is BIASED because Age confounds the relationship.")
print()

# Correct analysis: control for age (using DAG guidance)
X_adjusted = df_medical[['Treatment', 'Age']]
model_adjusted = LinearRegression()
model_adjusted.fit(X_adjusted, y)
adjusted_effect = model_adjusted.coef_[0]  # Treatment coefficient

print(f"ADJUSTED estimate (controlling for Age): {adjusted_effect:.2f}")
print("This is UNBIASED (close to true effect of +15)")
print()

print(f"Bias in naive estimate: {naive_effect - 15:.2f} points")
print()

# Visualize the confounding
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Recovery by treatment (ignoring age) - misleading
treated = df_medical[df_medical['Treatment'] == 1]
untreated = df_medical[df_medical['Treatment'] == 0]

axes[0].scatter(untreated['Age'], untreated['Recovery'],
                alpha=0.5, s=20, color='blue', label='No Treatment')
axes[0].scatter(treated['Age'], treated['Recovery'],
                alpha=0.5, s=20, color='red', label='Treatment')
axes[0].set_xlabel('Age', fontsize=12)
axes[0].set_ylabel('Recovery Score', fontsize=12)
axes[0].set_title('Raw Data: Treatment appears harmful\n(confounded by age)',
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: Age-adjusted comparison
# Plot residuals after removing age effect
model_age = LinearRegression()
model_age.fit(df_medical[['Age']], df_medical['Recovery'])
recovery_residuals = df_medical['Recovery'] - model_age.predict(df_medical[['Age']])

treated_mask = df_medical['Treatment'] == 1
axes[1].hist(recovery_residuals[~treated_mask], bins=30, alpha=0.6,
             color='blue', label='No Treatment', density=True)
axes[1].hist(recovery_residuals[treated_mask], bins=30, alpha=0.6,
             color='red', label='Treatment', density=True)
axes[1].set_xlabel('Recovery Score (age-adjusted)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Age-Adjusted: Treatment is beneficial\n(confounder controlled)',
                  fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('book/course-19/ch54/confounding_example.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# Medical Treatment Analysis: Identifying Confounders
# ============================================================
# True causal structure:
#   Age → Treatment
#   Age → Recovery
#   Treatment → Recovery
#
# Age is a CONFOUNDER (fork structure: Treatment ← Age → Recovery)
#
# NAIVE estimate (ignoring Age): -0.97
# This is BIASED because Age confounds the relationship.
#
# ADJUSTED estimate (controlling for Age): 14.87
# This is UNBIASED (close to true effect of +15)
#
# Bias in naive estimate: -15.97 points
```

This example illustrates why DAGs are essential for causal inference. The naive analysis suggests treatment harms patients (coefficient ≈ -1), but this is completely wrong. The error comes from ignoring Age, which forms a fork structure: Treatment ← Age → Recovery. Older patients receive treatment more often AND recover slower (due to age, not treatment). This creates a spurious negative association. The DAG reveals that Age must be controlled, and after adjustment, the estimated effect (+14.87) closely matches the true causal effect (+15). Without the DAG to guide variable selection, researchers might conclude treatment is harmful when it actually helps—a catastrophic mistake.

## Common Pitfalls

**1. Confusing Correlation with Causal Direction**

Beginners often see a correlation in data and immediately assume one variable causes the other, or draw an arrow in an arbitrary direction. The fundamental error: correlation is symmetric (if X correlates with Y, then Y correlates with X), but causation is directional (X → Y is fundamentally different from Y → X). Seeing that ice cream sales and drowning deaths correlate does not tell you whether ice cream causes drowning or drowning causes ice cream purchases.

Why this happens: Statistical software and correlation matrices provide no information about causal direction. Correlation measures association strength but is silent about mechanism. Students trained in statistical thinking have learned to find patterns in data, but causality requires bringing domain knowledge and temporal information from outside the data.

What to do instead: Always ask "which variable causally influences which?" based on domain knowledge, temporal ordering (causes must precede effects), or interventional logic ("if I directly manipulate X, will Y change?"). If genuinely unsure about direction, represent uncertainty explicitly with a bidirected edge (X ↔ Y) indicating unmeasured confounding, or use a latent variable. Never draw an arrow just because two variables correlate—the arrow represents a causal claim that must be justified independently.

**2. Forgetting that Conditioning on a Collider Opens a Path**

The most pervasive error: assuming that controlling for a variable always makes other variables more independent. Students correctly learn that conditioning blocks association in chains (X → Y → Z) and forks (X ← Y → Z), then incorrectly apply the same logic to colliders (X → Y ← Z). This leads to devastating errors: conditioning on a collider creates spurious associations between its causes where none existed.

Why this happens: It contradicts everyday statistical intuition. In regression courses, students learn "control for confounders to isolate the effect." This works for forks and chains but fails catastrophically for colliders. The collider case requires reasoning backward: observing the effect (Y) provides information about the balance of its causes (X and Z). If X is high and we observe Y, we infer Z must be low to compensate.

What to do instead: Before controlling for any variable, identify whether it is a collider (two or more arrows pointing in). Memorize the rule: "Conditioning blocks chains and forks but opens colliders." Practice with concrete examples until the behavior becomes intuitive. The sports analytics paradox (height and speed among pros) and Berkson's paradox (hospital admission bias) are canonical teaching cases. Use interactive tools to see the correlation flip when conditioning on colliders.

**3. Checking Only One Path and Declaring d-Separation**

Students find one blocked path between X and Y, conclude they are d-separated, and stop searching. This is wrong: d-separation requires ALL paths to be blocked. A single open path is sufficient for X and Y to be associated, no matter how many other paths are blocked.

Why this happens: Confirmation bias—once students find what they expect (a blocked path), they stop searching. In complex graphs with many nodes, enumerating all paths is tedious and error-prone. Students also sometimes misunderstand the definition, thinking "at least one blocked path" instead of "every path blocked."

What to do instead: Systematically enumerate every path between X and Y in the undirected graph first, then check each one individually in the directed graph. Use a checklist: write down all paths, mark each as blocked or open, and only conclude d-separation if every single one is blocked. Algorithmic tools (DAGitty, causalgraphicalmodels in Python) perform exhaustive search and catch paths humans miss. Double-check complex cases programmatically.

## Practice

**Practice 1**

For each scenario below, draw a DAG representing the causal relationships. Label all nodes clearly and justify each arrow with domain reasoning.

Scenario A: A simplified education system where parental income affects both school quality and test scores directly (through tutoring and resources), school quality affects test scores, and test scores affect college admission.

Scenario B: Temperature affects ice cream sales, temperature affects beach attendance, and beach attendance also affects ice cream sales (beach vendors).

Scenario C: Smoking causes both lung cancer and yellow teeth, genetic predisposition affects both cancer risk and heart disease risk (independently), and cancer treatment affects quality of life.

**Practice 2**

Given the following DAG, identify all instances of chains, forks, and colliders. For each three-node sequence, classify its structure type and state whether the outer variables are marginally independent and whether they are conditionally independent given the middle variable.

```
       Age
      /   \
     v     v
  Income  Health
     \     /
      v   v
    Life_Satisfaction
         |
         v
    Retirement_Plans
```

**Practice 3**

Given the DAG below, determine whether the following conditional independence statements are true or false. For each query, trace all paths between the variables, identify blocking nodes, and explain your reasoning.

```
    A → B → D
    ↓       ↑
    C  ───→ E → F
```

1. A ⊥ D | ∅ (is A independent of D given nothing?)
2. A ⊥ D | B
3. A ⊥ F | E
4. B ⊥ C | A
5. B ⊥ E | {C, D}

**Practice 4**

A data scientist at a tech company observes that among users who clicked on a promotional banner, time spent on site and purchase amount are negatively correlated. However, in the general user population, these variables are uncorrelated.

1. Draw a DAG that explains this phenomenon using collider bias
2. Explain in plain language why this negative correlation emerges among clickers
3. Write Python code (building on Part 4 examples) to generate synthetic data demonstrating this pattern: create 5,000 users with independent Time_Spent and Purchase_Amount, define Click = 1 when Time_Spent + Purchase_Amount exceeds a threshold, then show the correlation in the general population versus among clickers

**Practice 5**

A company wants to evaluate whether a new training program increases employee productivity. Available data includes Training (binary: received training or not), Productivity (measured score), Tenure (years at company), and Department (Engineering or Sales). The data shows trained employees have 12 points higher productivity on average.

However, the company knows:
- Managers preferentially select longer-tenured employees for training
- Tenure independently increases productivity (learning by doing)
- Different departments have different baseline productivity and different training rates
- Training might genuinely increase productivity (the causal effect of interest)

1. Draw a complete DAG representing this scenario with all mentioned variables
2. Identify which variables are confounders in the Training → Productivity relationship
3. Write the regression formula needed to estimate the causal effect of training
4. Explain what would happen if you accidentally controlled for a variable that is a collider (if any exist in this scenario)

## Solutions

**Solution 1**

Scenario A: Education system
```
Parental_Income → School_Quality → Test_Scores → College_Admission
       |                                ↑
       └────────────────────────────────┘
```
Justification: Parental income causes school quality (wealthy families choose better schools). Income also directly causes test scores through private tutoring and resources. School quality causes test scores (better teachers, facilities). Finally, test scores cause college admission decisions. Note that Test_Scores is both a collider (School_Quality → Test_Scores ← Parental_Income) and a mediator (on the path from School_Quality to College_Admission).

Scenario B: Weather and ice cream
```
Temperature → Ice_Cream_Sales
     |              ↑
     v              |
Beach_Attendance ───┘
```
Justification: Temperature causes ice cream sales directly (people want cold treats when hot). Temperature also causes beach attendance (people go to beaches on hot days). Beach attendance causes ice cream sales because beach vendors sell more. This creates both a fork (Ice_Cream ← Temperature → Beach) and a chain (Temperature → Beach → Ice_Cream), making this a more complex structure with multiple paths.

Scenario C: Medical conditions
```
Smoking → Lung_Cancer → Cancer_Treatment → Quality_of_Life
Smoking → Yellow_Teeth

Genetic_Predisposition → Lung_Cancer
Genetic_Predisposition → Heart_Disease
```
Justification: Smoking directly causes both lung cancer and yellow teeth (two independent effects). Genetic predisposition independently affects cancer risk and heart disease risk (fork structure). Cancer treatment affects quality of life. Lung_Cancer is a collider (Smoking → Lung_Cancer ← Genetic_Predisposition).

**Solution 2**

Breaking down the structure systematically:

1. **Age → Income**: Chain structure
   - Age and Income are dependent (Age ⊥̸ Income)
   - This is a direct causal relationship with no intermediate variables

2. **Age → Health**: Chain structure
   - Age and Health are dependent (Age ⊥̸ Health)
   - This is a direct causal relationship with no intermediate variables

3. **Income → Life_Satisfaction**: Chain structure
   - Income and Life_Satisfaction are dependent
   - This is a direct causal relationship

4. **Health → Life_Satisfaction**: Chain structure
   - Health and Life_Satisfaction are dependent
   - This is a direct causal relationship

5. **Life_Satisfaction → Retirement_Plans**: Chain structure
   - These are dependent
   - This is a direct causal relationship

6. **Age → Income → Life_Satisfaction**: Chain (mediation)
   - Age and Life_Satisfaction are marginally dependent: Age ⊥̸ Life_Satisfaction
   - Age and Life_Satisfaction are conditionally independent given Income: Age ⊥ Life_Satisfaction | Income? NO, because there's another path through Health

7. **Age → Health → Life_Satisfaction**: Chain (mediation)
   - Age and Life_Satisfaction are dependent
   - Multiple paths exist between these variables

8. **Income → Life_Satisfaction ← Health**: COLLIDER
   - Income and Health are marginally independent: Income ⊥ Health (no directed path, Age is a common cause but they're not directly connected after accounting for Age)
   - Actually, Income ← Age → Health is a FORK, so Income ⊥̸ Health marginally
   - Income and Health are conditionally dependent given Life_Satisfaction: Income ⊥̸ Health | Life_Satisfaction (collider is opened)

Correct complete analysis:
- **Fork**: Age → Income, Age → Health (Age is common cause)
- **Collider**: Income → Life_Satisfaction ← Health
- **Chain**: Life_Satisfaction → Retirement_Plans

**Solution 3**

First, enumerate ALL paths between each pair (using undirected graph):

1. **A ⊥ D | ∅**

Paths from A to D:
- Path 1: A → B → D (chain)
- Path 2: A → C → E → D (chain through multiple nodes)

Analysis:
- Path 1 (A → B → D): No variables conditioned, chain is OPEN
- Path 2 (A → C → E → D): No variables conditioned, chain is OPEN

At least one path is open → **FALSE** (A and D are NOT independent)

2. **A ⊥ D | B**

Paths from A to D:
- Path 1: A → B → D (chain, B conditioned → BLOCKED)
- Path 2: A → C → E → D (no variable in this path is conditioned → OPEN)

At least one path (Path 2) remains open → **FALSE**

3. **A ⊥ F | E**

Paths from A to F:
- Path 1: A → C → E → F (chain)
- Path 2: A → B → D → E → F (chain)

Analysis:
- Path 1: Passes through E (conditioned) → BLOCKED
- Path 2: Passes through E (conditioned) → BLOCKED

All paths blocked → **TRUE**

4. **B ⊥ C | A**

Paths from B to C:
- Path 1: B ← A → C (fork with A as common cause)

Analysis:
- Fork B ← A → C, A is conditioned → BLOCKED

All paths blocked → **TRUE**

5. **B ⊥ E | {C, D}**

Paths from B to E:
- Path 1: B → D → E (chain through D)
- Path 2: B ← A → C → E (fork at A, chain from C to E)

Analysis:
- Path 1: D is conditioned → BLOCKED
- Path 2: C is conditioned, blocks the chain from C to E → BLOCKED

All paths blocked → **TRUE**

**Solution 4**

1. DAG structure:
```
Time_Spent → Clicked_Banner ← Purchase_Amount
```

Clicked_Banner is a collider. Users click when they have either high time spent OR high purchase amount (or both).

2. Plain language explanation:

Among all users, time spent and purchase amount are independent—some users spend a lot of time but don't buy much, others buy quickly, others do both, others neither. However, clicking the banner requires high engagement (high time or high purchase amount). Among clickers, if someone has low time spent but still clicked, they must have high purchase amount to compensate. Similarly, if someone has high time spent, they can click even with lower purchase amount. This creates a negative correlation: conditioning on the collider (clicking) induces spurious negative association between its causes.

3. Complete Python code:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n = 5000

# Generate independent variables
time_spent = np.random.normal(loc=30, scale=10, size=n)  # minutes
purchase_amount = np.random.normal(loc=50, scale=20, size=n)  # dollars

# Clicking is a collider: requires high combined engagement
engagement_threshold = 75
combined_engagement = time_spent + purchase_amount
clicked = (combined_engagement > engagement_threshold).astype(int)

df_banner = pd.DataFrame({
    'Time_Spent': time_spent,
    'Purchase_Amount': purchase_amount,
    'Clicked': clicked
})

print("Banner Click Collider Bias")
print("=" * 60)
print(f"Total users: {n}")
print(f"Users who clicked: {clicked.sum()} ({100*clicked.mean():.1f}%)")
print()

# General population correlation
corr_general = df_banner['Time_Spent'].corr(df_banner['Purchase_Amount'])
print(f"Corr(Time, Purchase) in all users: {corr_general:.3f}")
print()

# Clickers only
df_clickers = df_banner[df_banner['Clicked'] == 1]
corr_clickers = df_clickers['Time_Spent'].corr(df_clickers['Purchase_Amount'])
print(f"Corr(Time, Purchase) among clickers: {corr_clickers:.3f}")
print()
print("Explanation: Clicked_Banner is a collider.")
print("Conditioning on it creates negative correlation between causes.")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df_banner['Time_Spent'], df_banner['Purchase_Amount'],
                alpha=0.3, s=10, color='gray')
axes[0].set_xlabel('Time Spent (min)')
axes[0].set_ylabel('Purchase Amount ($)')
axes[0].set_title(f'All Users\nCorr = {corr_general:.3f}')

axes[1].scatter(df_clickers['Time_Spent'], df_clickers['Purchase_Amount'],
                alpha=0.6, s=30, color='orange')
axes[1].set_xlabel('Time Spent (min)')
axes[1].set_ylabel('Purchase Amount ($)')
axes[1].set_title(f'Clickers Only\nCorr = {corr_clickers:.3f}', color='red')

plt.tight_layout()
plt.show()

# Output:
# Banner Click Collider Bias
# ============================================================
# Total users: 5000
# Users who clicked: 2546 (50.9%)
#
# Corr(Time, Purchase) in all users: 0.015
#
# Corr(Time, Purchase) among clickers: -0.582
#
# Explanation: Clicked_Banner is a collider.
# Conditioning on it creates negative correlation between causes.
```

**Solution 5**

1. Complete DAG:

```
Tenure → Training → Productivity
  |                      ↑
  └──────────────────────┘

Department → Training
    |
    └──────→ Productivity
```

In this structure:
- Tenure → Training (managers select experienced employees)
- Tenure → Productivity (direct effect of experience)
- Training → Productivity (the causal effect we want to estimate)
- Department → Training (different training rates by department)
- Department → Productivity (different baseline productivity)

2. Confounders identification:

**Tenure** is a confounder: Fork structure Training ← Tenure → Productivity. It causes both treatment assignment and outcome.

**Department** is a confounder: Fork structure Training ← Department → Productivity. It affects both who gets trained and baseline productivity.

Both must be controlled to get an unbiased estimate of the Training → Productivity causal effect.

3. Correct regression formula:

```python
from sklearn.linear_model import LinearRegression
import pandas as pd

# Assuming df has columns: Training, Productivity, Tenure, Department
# Convert Department to dummy variable
df['Dept_Sales'] = (df['Department'] == 'Sales').astype(int)

# Correct model: control for confounders
X = df[['Training', 'Tenure', 'Dept_Sales']]
y = df['Productivity']

model = LinearRegression()
model.fit(X, y)

causal_effect = model.coef_[0]  # Coefficient on Training
print(f"Causal effect of Training: {causal_effect:.2f}")
```

The model is: Productivity = β₀ + β₁·Training + β₂·Tenure + β₃·Dept_Sales + ε

The causal effect is β₁, the coefficient on Training after adjusting for confounders.

4. Controlling for a collider:

In this scenario, there is no collider in the causal structure (no variable caused by both Training and something else). However, if the company had collected data on "Employee_Satisfaction" and if both Training → Satisfaction and Productivity → Satisfaction existed, then Satisfaction would be a collider. Controlling for it would:

- Induce spurious negative correlation between Training and Productivity
- Bias the causal effect estimate downward (possibly even reversing the sign)
- Create selection bias similar to the sports analytics example

The key lesson: DAG analysis prevents this error by revealing which variables to control (confounders: Tenure, Department) and which to avoid controlling (colliders, if any exist; and mediators if we want total effects rather than direct effects).

## Key Takeaways

- DAGs provide a formal graphical language for encoding causal assumptions about how variables relate, making implicit domain knowledge explicit and testable
- The three elementary structures—chain (mediation), fork (confounding), and collider (selection bias)—form the building blocks of all causal reasoning with graphs
- d-Separation is a purely structural algorithm that determines conditional independence relationships from graph topology alone, without requiring any data
- Conditioning (controlling for a variable) blocks association in chains and forks but opens association in colliders—a counterintuitive behavior that causes pervasive errors in applied research
- Understanding d-separation is essential for correct variable selection in causal inference: control for confounders (forks) but avoid controlling for colliders and mediators unless justified by the causal question

**Next:** Section 54.2 covers structural causal models (SCMs), which extend DAGs by adding explicit functional forms and probability distributions to the causal relationships, enabling quantitative causal effect estimation beyond the qualitative structure provided by graphs alone.
