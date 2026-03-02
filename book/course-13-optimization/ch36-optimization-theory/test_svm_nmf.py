#!/usr/bin/env python3
"""
Test SVM and NMF code blocks
"""

import sys
import traceback
import numpy as np


def test_svm_dual():
    """Test SVM dual formulation with KKT conditions"""
    print("\n=== Testing SVM Dual Formulation ===")
    try:
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt
        import cvxopt
        cvxopt.solvers.options['show_progress'] = False

        # Load binary classification problem (2 classes, 2 features)
        iris = load_iris()
        # Use only setosa (0) and versicolor (1), and only first 2 features
        X = iris.data[:100, :2]
        y = iris.target[:100]
        y = np.where(y == 0, -1, 1)  # Convert to {-1, +1}

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        n = len(y)

        print("=== SVM Dual Formulation ===")
        print(f"Data: {n} samples, 2 features")
        print(f"Classes: {np.unique(y)} (counts: {np.bincount(y[y>0])})")

        # Solve SVM dual problem:
        # maximize: Σ αᵢ - (1/2) ΣΣ αᵢαⱼyᵢyⱼ⟨xᵢ,xⱼ⟩
        # subject to: αᵢ ≥ 0, Σ αᵢyᵢ = 0
        #
        # Convert to cvxopt format: minimize (1/2)xᵀPx + qᵀx
        # P = yᵢyⱼ⟨xᵢ,xⱼ⟩ (n×n matrix)
        # q = -1 (all ones, negated)

        # Construct kernel matrix K[i,j] = xᵢ·xⱼ
        K = X @ X.T

        # Construct P matrix for QP
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n))

        # Constraint: Σ αᵢyᵢ = 0
        A = cvxopt.matrix(y.reshape(1, -1).astype(float))
        b = cvxopt.matrix(0.0)

        # Constraint: αᵢ ≥ 0 (written as -αᵢ ≤ 0)
        G = cvxopt.matrix(-np.eye(n))
        h = cvxopt.matrix(np.zeros(n))

        # Solve QP
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.array(solution['x']).flatten()

        # Identify support vectors (α > threshold)
        threshold = 1e-5
        support_vectors = alpha > threshold
        sv_indices = np.where(support_vectors)[0]

        print(f"\n=== Solution ===")
        print(f"Number of support vectors: {np.sum(support_vectors)} / {n}")
        print(f"Support vector indices: {sv_indices[:10]}...")  # Show first 10

        # Compute weight vector: w = Σ αᵢyᵢxᵢ
        w = np.sum((alpha * y)[:, np.newaxis] * X, axis=0)
        print(f"Weight vector w: {w}")

        # Compute bias b using support vectors
        # For support vector i: yᵢ(w·xᵢ + b) = 1
        # => b = yᵢ - w·xᵢ
        b = np.mean(y[support_vectors] - X[support_vectors] @ w)
        print(f"Bias b: {b:.4f}")

        # Verify KKT conditions
        print("\n=== KKT Conditions Verification ===")

        # 1. Stationarity: ∇L = 0 (implicitly satisfied by QP solver)
        print("1. Stationarity: Satisfied by QP solver")

        # 2. Primal feasibility: yᵢ(w·xᵢ + b) ≥ 1
        margins = y * (X @ w + b)
        primal_feasible = np.all(margins >= 1 - 1e-6)
        print(f"2. Primal feasibility: {primal_feasible} (min margin: {margins.min():.4f})")

        # 3. Dual feasibility: αᵢ ≥ 0
        dual_feasible = np.all(alpha >= -1e-6)
        print(f"3. Dual feasibility: {dual_feasible} (min α: {alpha.min():.6f})")

        # 4. Complementary slackness: αᵢ(yᵢ(w·xᵢ + b) - 1) = 0
        complementary_slackness = alpha * (margins - 1)
        cs_satisfied = np.all(np.abs(complementary_slackness) < 1e-4)
        print(f"4. Complementary slackness: {cs_satisfied}")
        print(f"   Max violation: {np.abs(complementary_slackness).max():.6f}")

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: Decision boundary and support vectors
        ax = axes[0]
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                              np.linspace(y_min, y_max, 200))
        Z = (np.c_[xx.ravel(), yy.ravel()] @ w + b).reshape(xx.shape)

        # Plot decision boundary and margins
        ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2, linestyles='-')
        ax.contour(xx, yy, Z, levels=[-1, 1], colors='black', linewidths=1, linestyles='--', alpha=0.5)

        # Plot data points
        scatter1 = ax.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', s=50,
                              edgecolors='k', label='Class -1')
        scatter2 = ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=50,
                              edgecolors='k', label='Class +1')

        # Highlight support vectors
        ax.scatter(X[support_vectors, 0], X[support_vectors, 1],
                   s=200, facecolors='none', edgecolors='green', linewidths=3,
                   label=f'Support Vectors (n={np.sum(support_vectors)})')

        ax.set_xlabel('Feature 1 (standardized)', fontsize=11)
        ax.set_ylabel('Feature 2 (standardized)', fontsize=11)
        ax.set_title('SVM Decision Boundary and Support Vectors', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Right plot: Alpha values
        ax = axes[1]
        ax.bar(range(n), alpha, color=['green' if sv else 'gray' for sv in support_vectors])
        ax.axhline(threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold ({threshold})')
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('α (Lagrange Multiplier)', fontsize=11)
        ax.set_title('Lagrange Multipliers (α values)\nGreen = Support Vectors', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('diagrams/svm_dual.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("\n✓ SVM Dual PASSED")
        return True
    except Exception as e:
        print(f"✗ SVM Dual FAILED: {e}")
        traceback.print_exc()
        return False


def test_nmf():
    """Test NMF with projected gradient descent"""
    print("\n=== Testing NMF with Projected Gradient Descent ===")
    try:
        from sklearn.datasets import load_digits
        import matplotlib.pyplot as plt

        # Load digits dataset (8×8 grayscale images)
        digits = load_digits()
        X_images = digits.data[:100]  # Use first 100 images
        n_samples, n_features = X_images.shape
        n_components = 10  # Number of basis vectors to learn

        print("=== Non-Negative Matrix Factorization ===")
        print(f"Data: {n_samples} images, {n_features} pixels each")
        print(f"Decomposition: V ≈ WH")
        print(f"  V: {n_samples}×{n_features} (data matrix)")
        print(f"  W: {n_samples}×{n_components} (coefficients)")
        print(f"  H: {n_components}×{n_features} (basis vectors)")

        # Initialize W and H with small positive values
        np.random.seed(42)
        W = np.abs(np.random.randn(n_samples, n_components) * 0.1)
        H = np.abs(np.random.randn(n_components, n_features) * 0.1)

        def project_nonnegative(X):
            """Project onto non-negative orthant: set negative values to 0."""
            return np.maximum(X, 0)

        def reconstruction_error(V, W, H):
            """Frobenius norm of reconstruction error."""
            return np.linalg.norm(V - W @ H, 'fro')

        # Projected gradient descent
        learning_rate = 0.001
        n_iterations = 500
        errors = []

        print("\n=== Training with Projected Gradient Descent ===")

        for iteration in range(n_iterations):
            # Compute reconstruction and error
            V_recon = W @ H
            error = reconstruction_error(X_images, W, H)
            errors.append(error)

            if iteration % 100 == 0:
                print(f"Iteration {iteration:3d}, Error: {error:.2f}")

            # Gradient descent step for W
            # ∂||V - WH||²/∂W = 2(WH - V)Hᵀ
            grad_W = 2 * (V_recon - X_images) @ H.T
            W = W - learning_rate * grad_W
            W = project_nonnegative(W)  # Project to enforce W ≥ 0

            # Gradient descent step for H
            # ∂||V - WH||²/∂H = 2Wᵀ(WH - V)
            grad_H = 2 * W.T @ (V_recon - X_images)
            H = H - learning_rate * grad_H
            H = project_nonnegative(H)  # Project to enforce H ≥ 0

        final_error = errors[-1]
        print(f"Final error: {final_error:.2f}")

        # Verify non-negativity constraint is satisfied
        print(f"\n=== Constraint Verification ===")
        print(f"W min value: {W.min():.6f} (should be ≥ 0)")
        print(f"H min value: {H.min():.6f} (should be ≥ 0)")
        print(f"Non-negativity satisfied: {W.min() >= 0 and H.min() >= 0}")

        # Compare with sklearn's NMF
        from sklearn.decomposition import NMF
        nmf_sklearn = NMF(n_components=n_components, init='random', random_state=42, max_iter=500)
        W_sklearn = nmf_sklearn.fit_transform(X_images)
        H_sklearn = nmf_sklearn.components_
        error_sklearn = reconstruction_error(X_images, W_sklearn, H_sklearn)
        print(f"\nsklearn NMF error: {error_sklearn:.2f}")
        print(f"Our implementation error: {final_error:.2f}")

        # Visualization
        fig = plt.figure(figsize=(16, 10))

        # Plot 1: Convergence curve
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(errors, 'b-', linewidth=2)
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Reconstruction Error', fontsize=11)
        ax1.set_title('NMF Convergence with Projected Gradient Descent', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Learned basis vectors (components)
        n_show = 10
        for i in range(n_show):
            ax = plt.subplot(3, n_show, n_show + i + 1)
            ax.imshow(H[i].reshape(8, 8), cmap='gray', interpolation='nearest')
            ax.set_title(f'Basis {i+1}', fontsize=9)
            ax.axis('off')

        # Plot 3: Original vs. reconstructed images
        n_examples = 10
        for i in range(n_examples):
            # Original
            ax = plt.subplot(3, n_examples*2, 2*n_show + 2*i + 1)
            ax.imshow(X_images[i].reshape(8, 8), cmap='gray', interpolation='nearest')
            if i == 0:
                ax.set_ylabel('Original', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

            # Reconstructed
            ax = plt.subplot(3, n_examples*2, 2*n_show + 2*i + 2)
            reconstructed = (W[i:i+1] @ H).reshape(8, 8)
            ax.imshow(reconstructed, cmap='gray', interpolation='nearest')
            if i == 0:
                ax.set_ylabel('Reconstructed', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle('Non-Negative Matrix Factorization Results', fontsize=14, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig('diagrams/nmf_projected_gd.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("\n✓ NMF PASSED")
        return True
    except Exception as e:
        print(f"✗ NMF FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("TESTING SVM AND NMF CODE BLOCKS")
    print("=" * 80)

    results = []
    results.append(("SVM Dual Formulation", test_svm_dual()))
    results.append(("NMF Projected GD", test_nmf()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
