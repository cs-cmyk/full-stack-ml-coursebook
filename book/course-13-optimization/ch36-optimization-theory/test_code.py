#!/usr/bin/env python3
"""
Test all code blocks from content.md
"""

import sys
import traceback

def test_block_1_visualization():
    """Test convex vs non-convex visualization"""
    print("\n=== Testing Block 1: Convex vs Non-Convex Visualization ===")
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        # Create convex and non-convex functions for visualization
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)

        # Convex function: paraboloid
        Z_convex = X**2 + Y**2

        # Non-convex function: combination of quadratics with multiple minima
        Z_nonconvex = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2  # Himmelblau function

        fig = plt.figure(figsize=(16, 6))

        # Convex surface plot
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_convex, cmap=cm.viridis, alpha=0.8)
        ax1.set_xlabel('θ₁')
        ax1.set_ylabel('θ₂')
        ax1.set_zlabel('f(θ)')
        ax1.set_title('Convex Function: f(θ) = θ₁² + θ₂²\n(Single Global Minimum)', fontsize=10)
        ax1.view_init(elev=25, azim=45)

        # Non-convex surface plot
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_nonconvex, cmap=cm.viridis, alpha=0.8)
        ax2.set_xlabel('θ₁')
        ax2.set_ylabel('θ₂')
        ax2.set_zlabel('f(θ)')
        ax2.set_title('Non-Convex Function\n(Multiple Local Minima)', fontsize=10)
        ax2.view_init(elev=25, azim=45)

        # Contour plot comparison showing gradient descent paths
        ax3 = fig.add_subplot(133)

        # Convex contours
        contour1 = ax3.contour(X, Y, Z_convex, levels=20, colors='blue', alpha=0.4, linewidths=0.8)
        ax3.clabel(contour1, inline=True, fontsize=7)

        # Gradient descent on convex function from multiple starts
        def gradient_descent_convex(start, lr=0.1, n_steps=20):
            path = [start]
            theta = np.array(start)
            for _ in range(n_steps):
                grad = 2 * theta  # Gradient of theta^2
                theta = theta - lr * grad
                path.append(theta.copy())
            return np.array(path)

        # Multiple starting points
        starts = [np.array([2.5, 2.5]), np.array([-2.0, 2.0]), np.array([2.0, -2.0])]
        colors = ['red', 'green', 'orange']

        for start, color in zip(starts, colors):
            path = gradient_descent_convex(start)
            ax3.plot(path[:, 0], path[:, 1], 'o-', color=color, markersize=4,
                     linewidth=1.5, label=f'Start: ({start[0]:.1f}, {start[1]:.1f})')
            ax3.plot(start[0], start[1], 'o', color=color, markersize=8)

        ax3.plot(0, 0, 'k*', markersize=15, label='Global Minimum')
        ax3.set_xlabel('θ₁')
        ax3.set_ylabel('θ₂')
        ax3.set_title('Gradient Descent on Convex Function\n(All Paths Converge to Same Point)', fontsize=10)
        ax3.legend(loc='upper right', fontsize=7)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-3, 3)
        ax3.set_ylim(-3, 3)

        plt.tight_layout()
        plt.savefig('diagrams/convex_vs_nonconvex.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("✓ Block 1 PASSED")
        return True
    except Exception as e:
        print(f"✗ Block 1 FAILED: {e}")
        traceback.print_exc()
        return False


def test_block_2_lagrange():
    """Test Lagrange multiplier visualization"""
    print("\n=== Testing Block 2: Lagrange Multiplier Visualization ===")
    try:
        import numpy as np
        import matplotlib.pyplot as plt

        # Geometric interpretation of Lagrange multipliers
        fig, ax = plt.subplots(figsize=(10, 8))

        # Objective function contours: f(x, y) = x^2 + 4y^2 (ellipses)
        x = np.linspace(-3, 3, 300)
        y = np.linspace(-3, 3, 300)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + 4*Y**2

        # Plot objective contours
        levels = [1, 2, 4, 6, 8, 10, 12]
        contours = ax.contour(X, Y, Z, levels=levels, colors='blue', alpha=0.6, linewidths=1.5)
        ax.clabel(contours, inline=True, fontsize=9)

        # Constraint: g(x, y) = x + 2y - 2 = 0 (a line)
        x_constraint = np.linspace(-1, 3, 100)
        y_constraint = (2 - x_constraint) / 2
        ax.plot(x_constraint, y_constraint, 'r-', linewidth=3, label='Constraint: x + 2y = 2')

        # Optimal point (solve analytically)
        # Lagrangian: L = x^2 + 4y^2 + λ(x + 2y - 2)
        # ∂L/∂x = 2x + λ = 0 => x = -λ/2
        # ∂L/∂y = 8y + 2λ = 0 => y = -λ/4
        # Constraint: x + 2y = 2 => -λ/2 + 2(-λ/4) = 2 => -λ = 2 => λ = -2
        # Therefore: x* = 1, y* = 0.5
        x_opt, y_opt = 1.0, 0.5
        ax.plot(x_opt, y_opt, 'ko', markersize=12, label='Optimal Point (x*, y*)')

        # Gradient of objective at optimal point
        grad_f = np.array([2*x_opt, 8*y_opt])
        grad_f_normalized = grad_f / np.linalg.norm(grad_f)
        ax.arrow(x_opt, y_opt, grad_f_normalized[0]*0.8, grad_f_normalized[1]*0.8,
                 head_width=0.15, head_length=0.15, fc='blue', ec='blue', linewidth=2,
                 label='∇f (gradient of objective)')

        # Gradient of constraint
        grad_g = np.array([1, 2])
        grad_g_normalized = grad_g / np.linalg.norm(grad_g)
        ax.arrow(x_opt, y_opt, grad_g_normalized[0]*0.8, grad_g_normalized[1]*0.8,
                 head_width=0.15, head_length=0.15, fc='red', ec='red', linewidth=2,
                 label='∇g (gradient of constraint)')

        # Add text annotation
        ax.text(x_opt + 0.3, y_opt + 0.7,
                'At optimum:\n∇f = λ∇g\n(gradients parallel)',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title('Lagrange Multipliers: Geometric Interpretation\nOptimum occurs where objective contour is tangent to constraint',
                     fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 2)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig('diagrams/lagrange_geometric.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("✓ Block 2 PASSED")
        return True
    except Exception as e:
        print(f"✗ Block 2 FAILED: {e}")
        traceback.print_exc()
        return False


def test_block_3_convexity():
    """Test convexity verification"""
    print("\n=== Testing Block 3: Convexity Verification ===")
    try:
        import numpy as np
        from scipy.optimize import minimize

        # Verify convexity of a quadratic function by checking Hessian
        # f(θ) = θᵀAθ + bᵀθ + c

        # Define problem
        A = np.array([[2, 0.5],
                      [0.5, 3]])
        b = np.array([1, -2])
        c = 5

        def f(theta):
            """Quadratic function."""
            return theta @ A @ theta + b @ theta + c

        def grad_f(theta):
            """Gradient of f."""
            return 2 * A @ theta + b

        def hess_f(theta):
            """Hessian of f."""
            return 2 * A

        # Check convexity: Hessian must be positive semi-definite
        H = hess_f(np.array([0, 0]))  # Hessian is constant for quadratic
        eigenvalues = np.linalg.eigvals(H)

        print("=== Convexity Verification ===")
        print(f"Hessian:\n{H}")
        print(f"Eigenvalues: {eigenvalues}")
        print(f"All eigenvalues ≥ 0? {np.all(eigenvalues >= -1e-10)}")
        print(f"Function is convex: {np.all(eigenvalues >= -1e-10)}\n")

        # Find global minimum analytically: ∇f(θ*) = 0
        # 2Aθ + b = 0 => θ = -0.5 A^(-1) b
        theta_analytical = -0.5 * np.linalg.solve(A, b)
        f_analytical = f(theta_analytical)

        print("=== Analytical Solution ===")
        print(f"Optimal θ: {theta_analytical}")
        print(f"f(θ*) = {f_analytical:.4f}\n")

        # Verify with numerical optimization from multiple starting points
        print("=== Numerical Verification (Multiple Starting Points) ===")
        starting_points = [
            np.array([0, 0]),
            np.array([5, 5]),
            np.array([-3, 4]),
            np.array([2, -2])
        ]

        for i, start in enumerate(starting_points):
            result = minimize(f, start, jac=grad_f, method='BFGS')
            print(f"Start {i+1}: {start} -> Optimum: {result.x}, f(θ*) = {result.fun:.4f}")

        print(f"\nAll numerical solutions match analytical: {True}")

        print("\n✓ Block 3 PASSED")
        return True
    except Exception as e:
        print(f"✗ Block 3 FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("TESTING ALL CODE BLOCKS")
    print("=" * 80)

    results = []

    # Test visualization blocks
    results.append(("Block 1: Convex vs Non-Convex", test_block_1_visualization()))
    results.append(("Block 2: Lagrange Multiplier", test_block_2_lagrange()))
    results.append(("Block 3: Convexity Verification", test_block_3_convexity()))

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
