#!/usr/bin/env python3
"""
Test optimization method comparison and neural network landscape
"""

import sys
import traceback
import numpy as np


def test_gd_vs_lbfgs():
    """Test gradient descent vs L-BFGS comparison"""
    print("\n=== Testing GD vs L-BFGS Comparison ===")
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt
        from scipy.optimize import fmin_l_bfgs_b
        import time

        # Load breast cancer dataset (binary classification)
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        n_samples, n_features = X_train.shape

        print("=== Dataset ===")
        print(f"Training samples: {n_samples}, Features: {n_features}")
        print(f"Test samples: {len(y_test)}")

        # Logistic regression loss and gradient
        def sigmoid(z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

        def logistic_loss_and_grad(theta, X, y):
            """Compute loss and gradient for logistic regression."""
            z = X @ theta
            h = sigmoid(z)

            # Binary cross-entropy loss
            loss = -np.mean(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))

            # Gradient
            grad = X.T @ (h - y) / len(y)

            return loss, grad

        # Initialize parameters
        theta_init = np.zeros(n_features)

        # ========================================
        # Method 1: Gradient Descent (First-Order)
        # ========================================
        print("\n=== Method 1: Gradient Descent ===")

        theta_gd = theta_init.copy()
        learning_rate = 0.5
        n_iterations = 1000
        losses_gd = []
        times_gd = [0]

        start_time = time.time()
        for iteration in range(n_iterations):
            loss, grad = logistic_loss_and_grad(theta_gd, X_train, y_train)
            losses_gd.append(loss)
            theta_gd = theta_gd - learning_rate * grad
            times_gd.append(time.time() - start_time)

            if iteration % 200 == 0:
                print(f"  Iteration {iteration:4d}, Loss: {loss:.6f}")

        time_gd = time.time() - start_time
        final_loss_gd = losses_gd[-1]
        test_loss_gd = logistic_loss_and_grad(theta_gd, X_test, y_test)[0]

        print(f"  Final training loss: {final_loss_gd:.6f}")
        print(f"  Test loss: {test_loss_gd:.6f}")
        print(f"  Total time: {time_gd:.3f}s")
        print(f"  Iterations: {n_iterations}")

        # ========================================
        # Method 2: L-BFGS (Second-Order)
        # ========================================
        print("\n=== Method 2: L-BFGS ===")

        theta_lbfgs = theta_init.copy()
        losses_lbfgs = []
        times_lbfgs = [0]
        start_time = time.time()
        iteration_count = [0]

        def callback(theta):
            """Called after each L-BFGS iteration."""
            loss = logistic_loss_and_grad(theta, X_train, y_train)[0]
            losses_lbfgs.append(loss)
            times_lbfgs.append(time.time() - start_time)
            iteration_count[0] += 1

        def f_lbfgs(theta):
            """Wrapper for L-BFGS: returns loss and gradient."""
            return logistic_loss_and_grad(theta, X_train, y_train)

        # Run L-BFGS
        theta_lbfgs, min_loss, info_dict = fmin_l_bfgs_b(
            f_lbfgs,
            theta_init,
            maxiter=100,
            callback=callback,
            factr=1e7  # Convergence tolerance
        )

        time_lbfgs = time.time() - start_time
        final_loss_lbfgs = losses_lbfgs[-1]
        test_loss_lbfgs = logistic_loss_and_grad(theta_lbfgs, X_test, y_test)[0]

        print(f"  Final training loss: {final_loss_lbfgs:.6f}")
        print(f"  Test loss: {test_loss_lbfgs:.6f}")
        print(f"  Total time: {time_lbfgs:.3f}s")
        print(f"  Iterations: {iteration_count[0]}")
        print(f"  Function evaluations: {info_dict['funcalls']}")

        # ========================================
        # Comparison
        # ========================================
        print("\n=== Comparison ===")
        print(f"{'Method':<20} {'Iterations':<12} {'Time (s)':<12} {'Final Loss':<12} {'Test Loss':<12}")
        print("-" * 68)
        print(f"{'Gradient Descent':<20} {n_iterations:<12} {time_gd:<12.3f} {final_loss_gd:<12.6f} {test_loss_gd:<12.6f}")
        print(f"{'L-BFGS':<20} {iteration_count[0]:<12} {time_lbfgs:<12.3f} {final_loss_lbfgs:<12.6f} {test_loss_lbfgs:<12.6f}")

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Loss vs. Iterations
        ax = axes[0]
        ax.plot(range(len(losses_gd)), losses_gd, 'b-', linewidth=2, label='Gradient Descent')
        ax.plot(range(len(losses_lbfgs)), losses_lbfgs, 'r-', linewidth=2, label='L-BFGS')
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Training Loss', fontsize=11)
        ax.set_title('Convergence Comparison: Loss vs. Iterations', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 2: Loss vs. Wall-Clock Time
        ax = axes[1]
        ax.plot(times_gd, losses_gd, 'b-', linewidth=2, label='Gradient Descent')
        ax.plot(times_lbfgs, losses_lbfgs, 'r-', linewidth=2, label='L-BFGS')
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Training Loss', fontsize=11)
        ax.set_title('Convergence Comparison: Loss vs. Time', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig('diagrams/gd_vs_lbfgs.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("\n✓ GD vs L-BFGS PASSED")
        return True
    except Exception as e:
        print(f"✗ GD vs L-BFGS FAILED: {e}")
        traceback.print_exc()
        return False


def test_nn_landscape():
    """Test neural network loss landscape visualization"""
    print("\n=== Testing Neural Network Loss Landscape ===")
    try:
        from sklearn.datasets import make_moons
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Generate non-linear classification dataset
        X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to torch tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.LongTensor(y_test)

        print("=== Dataset ===")
        print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")

        # Define small neural network
        class SmallNet(nn.Module):
            def __init__(self):
                super(SmallNet, self).__init__()
                self.fc1 = nn.Linear(2, 8)
                self.fc2 = nn.Linear(8, 2)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # Train network and save trajectory
        torch.manual_seed(42)
        model = SmallNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        # Save parameter trajectory for visualization
        trajectory = []

        def get_params_flat():
            """Flatten all parameters into single vector."""
            return torch.cat([p.view(-1) for p in model.parameters()]).detach().numpy()

        print("\n=== Training Network ===")
        n_epochs = 100
        for epoch in range(n_epochs):
            # Forward pass
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save parameters every 10 epochs
            if epoch % 10 == 0:
                trajectory.append(get_params_flat())
                if epoch % 20 == 0:
                    print(f"  Epoch {epoch:3d}, Loss: {loss.item():.4f}")

        trajectory.append(get_params_flat())
        trajectory = np.array(trajectory)

        # Test accuracy
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test_t).sum().item() / len(y_test_t)
        print(f"\nTest accuracy: {accuracy:.4f}")

        # ========================================
        # Visualize Loss Landscape
        # ========================================
        print("\n=== Generating Loss Landscape ===")

        # Get final parameters
        theta_final = get_params_flat()

        # Generate two random orthonormal directions in parameter space
        np.random.seed(42)
        d1 = np.random.randn(len(theta_final))
        d1 = d1 / np.linalg.norm(d1)

        d2 = np.random.randn(len(theta_final))
        d2 = d2 - (d2 @ d1) * d1  # Orthogonalize
        d2 = d2 / np.linalg.norm(d2)

        def set_params_flat(params):
            """Set model parameters from flattened vector."""
            offset = 0
            for p in model.parameters():
                numel = p.numel()
                p.data = torch.FloatTensor(params[offset:offset + numel].reshape(p.shape))
                offset += numel

        def compute_loss(params):
            """Compute loss for given parameters."""
            set_params_flat(params)
            with torch.no_grad():
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
            return loss.item()

        # Create grid in parameter space
        alpha_range = np.linspace(-1.5, 1.5, 30)
        beta_range = np.linspace(-1.5, 1.5, 30)
        Alpha, Beta = np.meshgrid(alpha_range, beta_range)

        # Compute loss over grid
        Loss = np.zeros_like(Alpha)
        for i in range(len(alpha_range)):
            for j in range(len(beta_range)):
                # Parameters: theta_final + alpha*d1 + beta*d2
                params = theta_final + Alpha[j, i] * d1 + Beta[j, i] * d2
                Loss[j, i] = compute_loss(params)
            if i % 5 == 0:
                print(f"  Progress: {i+1}/{len(alpha_range)}")

        print("Loss landscape generated.")

        # Project trajectory onto 2D plane
        trajectory_2d = np.zeros((len(trajectory), 2))
        for i, params in enumerate(trajectory):
            diff = params - theta_final
            trajectory_2d[i, 0] = diff @ d1
            trajectory_2d[i, 1] = diff @ d2

        # Visualization
        fig = plt.figure(figsize=(16, 6))

        # 3D surface plot
        ax1 = fig.add_subplot(131, projection='3d')
        surf = ax1.plot_surface(Alpha, Beta, Loss, cmap='viridis', alpha=0.8,
                                 edgecolor='none', antialiased=True)
        ax1.plot(trajectory_2d[:, 0], trajectory_2d[:, 1],
                 [compute_loss(theta_final + t2d[0]*d1 + t2d[1]*d2) for t2d in trajectory_2d],
                 'r-o', linewidth=2, markersize=4, label='Training Path')
        ax1.set_xlabel('Direction 1', fontsize=10)
        ax1.set_ylabel('Direction 2', fontsize=10)
        ax1.set_zlabel('Loss', fontsize=10)
        ax1.set_title('Neural Network Loss Landscape\n(3D Surface)', fontsize=11)
        ax1.view_init(elev=25, azim=135)
        fig.colorbar(surf, ax=ax1, shrink=0.5)

        # Contour plot with trajectory
        ax2 = fig.add_subplot(132)
        contour = ax2.contour(Alpha, Beta, Loss, levels=20, cmap='viridis', linewidths=1)
        ax2.clabel(contour, inline=True, fontsize=7)
        ax2.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'r-o', linewidth=2,
                 markersize=5, label='Training Path')
        ax2.plot(0, 0, 'r*', markersize=15, label='Final Parameters')
        ax2.set_xlabel('Direction 1', fontsize=10)
        ax2.set_ylabel('Direction 2', fontsize=10)
        ax2.set_title('Loss Landscape Contours\nwith Training Trajectory', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Decision boundary
        ax3 = fig.add_subplot(133)
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                              np.linspace(y_min, y_max, 200))
        with torch.no_grad():
            Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
            Z = torch.softmax(Z, dim=1)[:, 1].reshape(xx.shape).numpy()

        ax3.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
        ax3.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                   c='blue', s=30, edgecolors='k', label='Class 0')
        ax3.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                   c='red', s=30, edgecolors='k', label='Class 1')
        ax3.set_xlabel('Feature 1', fontsize=10)
        ax3.set_ylabel('Feature 2', fontsize=10)
        ax3.set_title(f'Learned Decision Boundary\n(Test Acc: {accuracy:.3f})', fontsize=11)
        ax3.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig('diagrams/loss_landscape.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("\n✓ NN Loss Landscape PASSED")
        return True
    except Exception as e:
        print(f"✗ NN Loss Landscape FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("TESTING OPTIMIZATION METHODS")
    print("=" * 80)

    results = []
    results.append(("GD vs L-BFGS", test_gd_vs_lbfgs()))
    results.append(("NN Loss Landscape", test_nn_landscape()))

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
