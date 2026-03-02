"""
Code Review Test Script for Chapter 54
Tests all code blocks sequentially to verify they execute correctly
"""

import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Suppress matplotlib display
plt.ioff()

def test_block_1():
    """Visualization of the three elementary structures"""
    print("\n" + "="*60)
    print("Testing Block 1: Elementary Structures Visualization")
    print("="*60)

    try:
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
        plt.close()

        print("✓ Block 1 passed: Elementary structures visualized")
        return True
    except Exception as e:
        print(f"✗ Block 1 failed: {e}")
        traceback.print_exc()
        return False


def test_block_2():
    """Building a DAG representing a medical scenario"""
    print("\n" + "="*60)
    print("Testing Block 2: Medical DAG")
    print("="*60)

    try:
        # Create a directed graph
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
        plt.close()

        # Query graph structure programmatically
        print("Parents of Lung_Cancer:", list(G.predecessors('Lung_Cancer')))
        print("Children of Genetic_Factor:", list(G.successors('Genetic_Factor')))

        # Check for paths
        print("\nAll paths from Smoking to Chronic_Cough:")
        for path in nx.all_simple_paths(G.to_undirected(), 'Smoking', 'Chronic_Cough'):
            print("  ", path)

        print("✓ Block 2 passed: Medical DAG created")
        return True
    except Exception as e:
        print(f"✗ Block 2 failed: {e}")
        traceback.print_exc()
        return False


def test_block_3():
    """Generate synthetic data demonstrating each structure"""
    print("\n" + "="*60)
    print("Testing Block 3: Three Elementary Structures Data")
    print("="*60)

    try:
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

        print("✓ Block 3 passed: Three structures data generated")
        return True
    except Exception as e:
        print(f"✗ Block 3 failed: {e}")
        traceback.print_exc()
        return False


def test_block_4():
    """Testing d-separation with causalgraphicalmodels"""
    print("\n" + "="*60)
    print("Testing Block 4: d-Separation with causalgraphicalmodels")
    print("="*60)

    try:
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

        print("✓ Block 4 passed: d-Separation testing completed")
        return True
    except ImportError:
        print("✗ Block 4 failed: causalgraphicalmodels library not installed")
        print("  Install with: pip install causalgraphicalmodels")
        return False
    except Exception as e:
        print(f"✗ Block 4 failed: {e}")
        traceback.print_exc()
        return False


def test_block_5():
    """Collider Bias - Sports Analytics Paradox"""
    print("\n" + "="*60)
    print("Testing Block 5: Sports Analytics Collider Bias")
    print("="*60)

    try:
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
        plt.close()

        print("✓ Block 5 passed: Sports collider bias demonstrated")
        return True
    except Exception as e:
        print(f"✗ Block 5 failed: {e}")
        traceback.print_exc()
        return False


def test_block_6():
    """Using DAGs to Identify Confounders"""
    print("\n" + "="*60)
    print("Testing Block 6: Identifying Confounders")
    print("="*60)

    try:
        np.random.seed(42)
        n = 1000

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
        plt.close()

        print("✓ Block 6 passed: Confounder identification demonstrated")
        return True
    except Exception as e:
        print(f"✗ Block 6 failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests and generate report"""
    print("="*60)
    print("CODE REVIEW: Chapter 54 - DAGs and d-Separation")
    print("="*60)

    tests = [
        test_block_1,
        test_block_2,
        test_block_3,
        test_block_4,
        test_block_5,
        test_block_6
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
