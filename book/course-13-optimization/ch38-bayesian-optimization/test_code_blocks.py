"""
Code Review Test Script for Chapter 38: Bayesian Optimization
Tests all code blocks to verify they execute correctly.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CODE REVIEW: Chapter 38 - Bayesian Optimization")
print("="*80)

# Track results
results = []
total_blocks = 6
passed = 0
failed = 0

def test_block(block_num, description):
    """Test a code block and track results."""
    global passed, failed
    print(f"\n{'='*80}")
    print(f"Testing Block {block_num}: {description}")
    print(f"{'='*80}")
    try:
        return True
    except Exception as e:
        print(f"ERROR in Block {block_num}: {e}")
        results.append((block_num, description, "FAILED", str(e)))
        failed += 1
        return False

# =============================================================================
# BLOCK 1: The Expensive Black-Box Optimization Problem
# =============================================================================
if test_block(1, "Expensive Black-Box Optimization Problem"):
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import fetch_california_housing

        np.random.seed(42)

        # Load California Housing dataset
        housing = fetch_california_housing()
        X, y = housing.data, housing.target

        print("Dataset shape:", X.shape)
        print("Target shape:", y.shape)
        assert X.shape == (20640, 8), f"Expected X.shape (20640, 8), got {X.shape}"
        assert y.shape == (20640,), f"Expected y.shape (20640,), got {y.shape}"

        # Define expensive objective function
        def expensive_objective(n_estimators, max_depth):
            start_time = time.time()
            model = RandomForestRegressor(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth) if max_depth > 0 else None,
                random_state=42,
                n_jobs=-1
            )
            scores = cross_val_score(model, X, y, cv=3, scoring='r2')
            mean_score = scores.mean()
            elapsed = time.time() - start_time
            return mean_score, elapsed

        # Test evaluation
        print("\n--- Evaluating 3 random hyperparameter configurations ---")
        configs = [(50, 10), (100, 20), (200, 30)]

        test_results = []
        for n_est, depth in configs:
            score, time_taken = expensive_objective(n_est, depth)
            test_results.append((n_est, depth, score, time_taken))
            print(f"n_estimators={n_est:3d}, max_depth={depth:2d} → R²={score:.4f} (took {time_taken:.2f}s)")

        total_time = sum(r[3] for r in test_results)
        print(f"\nTotal time for 3 evaluations: {total_time:.2f}s")
        print(f"Grid search over 10×10 grid would take ~{(total_time/3)*100:.0f}s ({(total_time/3)*100/60:.1f} min)")

        results.append((1, "Expensive Black-Box Problem", "PASSED", "All checks passed"))
        passed += 1
        print("✓ Block 1 PASSED")
    except Exception as e:
        print(f"✗ Block 1 FAILED: {e}")
        results.append((1, "Expensive Black-Box Problem", "FAILED", str(e)))
        failed += 1

# =============================================================================
# BLOCK 2: Gaussian Process as Surrogate Model
# =============================================================================
if test_block(2, "Gaussian Process as Surrogate Model"):
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel

        # Create 1D test function
        def test_function_1d(x):
            return np.sin(3*x) + 0.3*np.cos(9*x) + 0.5*x

        # Generate domain
        x_domain = np.linspace(0, 5, 500).reshape(-1, 1)
        y_true = test_function_1d(x_domain)

        # Start with 5 random observations
        n_initial = 5
        np.random.seed(42)
        X_observed = np.random.uniform(0, 5, n_initial).reshape(-1, 1)
        y_observed = test_function_1d(X_observed).ravel()

        print(f"Initial observations: {n_initial} points")
        print("X_observed:", X_observed.ravel().round(2))
        print("y_observed:", y_observed.round(3))

        # Fit Gaussian Process
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, random_state=42)
        gp.fit(X_observed, y_observed)

        # Make predictions
        y_mean, y_std = gp.predict(x_domain, return_std=True)

        # Visualize
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x_domain, y_true, 'k-', linewidth=2, label='True function (unknown)', alpha=0.4)
        ax.plot(x_domain, y_mean, 'b-', linewidth=2, label='GP mean (μ)')
        ax.fill_between(x_domain.ravel(), y_mean - 2*y_std, y_mean + 2*y_std,
                        alpha=0.3, color='blue', label='95% confidence (μ ± 2σ)')
        ax.scatter(X_observed, y_observed, c='red', s=100, zorder=5,
                   marker='o', edgecolors='black', linewidths=1.5,
                   label='Observed points')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('f(x)', fontsize=12)
        ax.set_title('Gaussian Process Surrogate Model', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/home/chirag/ds-book/book/course-13/ch38/gp_surrogate.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("✓ GP surrogate plot saved")
        results.append((2, "GP Surrogate Model", "PASSED", "All checks passed"))
        passed += 1
        print("✓ Block 2 PASSED")
    except Exception as e:
        print(f"✗ Block 2 FAILED: {e}")
        results.append((2, "GP Surrogate Model", "FAILED", str(e)))
        failed += 1

# =============================================================================
# BLOCK 3: Acquisition Functions - Exploration vs. Exploitation
# =============================================================================
if test_block(3, "Acquisition Functions"):
    try:
        from scipy.stats import norm

        def expected_improvement(x, gp, y_best):
            mu, sigma = gp.predict(x, return_std=True)
            sigma = sigma.reshape(-1, 1)
            with np.errstate(divide='warn'):
                Z = (mu - y_best) / sigma
            ei = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            return ei.ravel()

        def upper_confidence_bound(x, gp, beta=2.0):
            mu, sigma = gp.predict(x, return_std=True)
            return mu + beta * sigma

        def thompson_sampling(x, gp, n_samples=1):
            y_samples = gp.sample_y(x, n_samples=n_samples, random_state=42)
            return y_samples.ravel()

        # Compute acquisition functions (reusing GP from block 2)
        y_best = y_observed.max()

        ei_values = expected_improvement(x_domain, gp, y_best)
        ucb_values = upper_confidence_bound(x_domain, gp, beta=2.0)
        ts_sample = thompson_sampling(x_domain, gp, n_samples=1)

        # Find next points
        next_x_ei = x_domain[np.argmax(ei_values)]
        next_x_ucb = x_domain[np.argmax(ucb_values)]
        next_x_ts = x_domain[np.argmax(ts_sample)]

        print(f"Current best value: f_best = {y_best:.3f}")
        print(f"\nNext point suggestions:")
        print(f"  Expected Improvement (EI): x = {next_x_ei[0]:.3f}")
        print(f"  Upper Confidence Bound (UCB): x = {next_x_ucb[0]:.3f}")
        print(f"  Thompson Sampling (TS): x = {next_x_ts[0]:.3f}")

        # Visualize acquisition functions
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))

        # Subplot 1: GP with observations
        axes[0].plot(x_domain, y_true, 'k-', linewidth=2, label='True function', alpha=0.4)
        axes[0].plot(x_domain, y_mean, 'b-', linewidth=2, label='GP mean')
        axes[0].fill_between(x_domain.ravel(), y_mean - 2*y_std, y_mean + 2*y_std,
                              alpha=0.3, color='blue')
        axes[0].scatter(X_observed, y_observed, c='red', s=100, zorder=5,
                        marker='o', edgecolors='black', linewidths=1.5)
        axes[0].axhline(y_best, color='red', linestyle='--', linewidth=1.5,
                        alpha=0.7, label=f'Best so far = {y_best:.3f}')
        axes[0].set_ylabel('f(x)', fontsize=11)
        axes[0].set_title('Gaussian Process Surrogate', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Subplot 2: Expected Improvement
        axes[1].plot(x_domain, ei_values, 'purple', linewidth=2.5)
        axes[1].axvline(next_x_ei, color='purple', linestyle='--', linewidth=2,
                        label=f'Next point: x={next_x_ei[0]:.3f}')
        axes[1].scatter([next_x_ei], [ei_values.max()], c='purple', s=200,
                        marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        axes[1].set_ylabel('EI(x)', fontsize=11)
        axes[1].set_title('Expected Improvement', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        # Subplot 3: Upper Confidence Bound
        axes[2].plot(x_domain, ucb_values, 'orange', linewidth=2.5)
        axes[2].axvline(next_x_ucb, color='orange', linestyle='--', linewidth=2,
                        label=f'Next point: x={next_x_ucb[0]:.3f}')
        axes[2].scatter([next_x_ucb], [ucb_values.max()], c='orange', s=200,
                        marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        axes[2].set_ylabel('UCB(x)', fontsize=11)
        axes[2].set_title('Upper Confidence Bound (β=2.0)', fontsize=13, fontweight='bold')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)

        # Subplot 4: Thompson Sampling
        axes[3].plot(x_domain, ts_sample, 'green', linewidth=2.5, label='Sampled function')
        axes[3].axvline(next_x_ts, color='green', linestyle='--', linewidth=2,
                        label=f'Next point: x={next_x_ts[0]:.3f}')
        axes[3].scatter([next_x_ts], [ts_sample.max()], c='green', s=200,
                        marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        axes[3].set_xlabel('x', fontsize=11)
        axes[3].set_ylabel('Sampled f(x)', fontsize=11)
        axes[3].set_title('Thompson Sampling', fontsize=13, fontweight='bold')
        axes[3].legend(fontsize=9)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/chirag/ds-book/book/course-13/ch38/acquisition_functions.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("✓ Acquisition functions plot saved")
        results.append((3, "Acquisition Functions", "PASSED", "All checks passed"))
        passed += 1
        print("✓ Block 3 PASSED")
    except Exception as e:
        print(f"✗ Block 3 FAILED: {e}")
        results.append((3, "Acquisition Functions", "FAILED", str(e)))
        failed += 1

# =============================================================================
# BLOCK 4: Complete Bayesian Optimization Loop
# =============================================================================
if test_block(4, "Complete Bayesian Optimization Loop"):
    try:
        from scipy.optimize import minimize

        # Define 2D Branin function
        def branin(x1, x2):
            a = 1.0
            b = 5.1 / (4 * np.pi**2)
            c = 5.0 / np.pi
            r = 6.0
            s = 10.0
            t = 1.0 / (8 * np.pi)
            term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
            term2 = s * (1 - t) * np.cos(x1)
            term3 = s
            return term1 + term2 + term3

        def branin_vectorized(X):
            return branin(X[:, 0], X[:, 1])

        def objective_2d(X):
            return -branin_vectorized(X)

        # Set up domain bounds
        bounds = np.array([[-5, 10], [0, 15]])

        # Initialize with random samples
        n_initial = 5
        np.random.seed(42)
        X_init = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(n_initial, 2))
        y_init = objective_2d(X_init)

        print("Initial samples:")
        for i in range(n_initial):
            print(f"  x1={X_init[i,0]:6.2f}, x2={X_init[i,1]:6.2f} → f={y_init[i]:7.4f}")

        # Bayesian Optimization Loop (reduced to 10 iterations for speed)
        n_iterations = 10
        X_observed = X_init.copy()
        y_observed = y_init.copy()

        history_best = []

        for iteration in range(n_iterations):
            # Fit GP
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                         n_restarts_optimizer=5, random_state=42)
            gp.fit(X_observed, y_observed)

            # Define acquisition function (EI)
            y_best = y_observed.max()

            def neg_ei(x):
                x = x.reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                if sigma == 0:
                    return 0
                Z = (mu - y_best) / sigma
                ei = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
                return -ei[0]

            # Optimize acquisition function
            best_ei = np.inf
            best_x = None

            for _ in range(10):
                x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
                result = minimize(neg_ei, x0,
                                bounds=[(bounds[0,0], bounds[0,1]), (bounds[1,0], bounds[1,1])],
                                method='L-BFGS-B')
                if result.fun < best_ei:
                    best_ei = result.fun
                    best_x = result.x

            # Evaluate objective
            x_next = best_x.reshape(1, -1)
            y_next = objective_2d(x_next)

            # Update dataset
            X_observed = np.vstack([X_observed, x_next])
            y_observed = np.append(y_observed, y_next)

            current_best = y_observed.max()
            history_best.append(current_best)

            if iteration >= n_iterations - 5:
                print(f"Iteration {iteration+1:2d}: x1={x_next[0,0]:6.2f}, x2={x_next[0,1]:6.2f} "
                      f"→ f={y_next[0]:7.4f} | Best so far: {current_best:7.4f}")

        print(f"\nOptimization complete!")
        print(f"Best value found: {y_observed.max():.6f}")
        print(f"True global minimum (negated): {-0.397887:.6f}")
        best_idx = np.argmax(y_observed)
        print(f"Best point: x1={X_observed[best_idx, 0]:.4f}, x2={X_observed[best_idx, 1]:.4f}")

        assert y_observed.max() > -1.0, "Should find reasonable optimum"

        results.append((4, "Complete BO Loop", "PASSED", "All checks passed"))
        passed += 1
        print("✓ Block 4 PASSED")
    except Exception as e:
        print(f"✗ Block 4 FAILED: {e}")
        results.append((4, "Complete BO Loop", "FAILED", str(e)))
        failed += 1

# =============================================================================
# BLOCK 5: Hyperparameter Tuning with Optuna
# =============================================================================
if test_block(5, "Hyperparameter Tuning with Optuna"):
    try:
        import optuna
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Load data
        housing = fetch_california_housing()
        X, y = housing.data, housing.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Training set:", X_train.shape)
        print("Test set:", X_test.shape)

        # Define objective
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': 42
            }

            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            return r2

        # Run Bayesian Optimization (reduced to 20 trials for speed)
        print("\n--- Running Bayesian Optimization with Optuna ---")
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=20, show_progress_bar=False)

        print(f"\nNumber of trials: {len(study.trials)}")
        print(f"Best R² score: {study.best_value:.6f}")
        print(f"\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Compare with Random Search
        print("\n--- Comparison: Random Search (20 trials) ---")
        study_random = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.RandomSampler(seed=42)
        )
        study_random.optimize(objective, n_trials=20, show_progress_bar=False)

        print(f"Best R² score (Random Search): {study_random.best_value:.6f}")
        print(f"Best R² score (Bayesian Opt):  {study.best_value:.6f}")
        improvement = (study.best_value - study_random.best_value)*100
        print(f"Difference: {improvement:.2f}%")

        # Visualize convergence
        fig, ax = plt.subplots(figsize=(12, 6))

        bo_best = [max(study.trials[i].value for i in range(j+1))
                   for j in range(len(study.trials))]
        rs_best = [max(study_random.trials[i].value for i in range(j+1))
                   for j in range(len(study_random.trials))]

        ax.plot(range(1, len(bo_best)+1), bo_best, 'b-', linewidth=2.5,
                marker='o', markersize=4, label='Bayesian Optimization (TPE)')
        ax.plot(range(1, len(rs_best)+1), rs_best, 'orange', linewidth=2.5,
                marker='s', markersize=4, label='Random Search', linestyle='--')

        ax.set_xlabel('Number of Trials', fontsize=12)
        ax.set_ylabel('Best R² Score Found', fontsize=12)
        ax.set_title('Convergence: Bayesian Optimization vs. Random Search',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/home/chirag/ds-book/book/course-13/ch38/bo_vs_random_convergence.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("✓ Convergence plot saved")

        assert study.best_value > 0.5, "Should achieve reasonable R² score"

        results.append((5, "Hyperparameter Tuning with Optuna", "PASSED", "All checks passed"))
        passed += 1
        print("✓ Block 5 PASSED")
    except Exception as e:
        print(f"✗ Block 5 FAILED: {e}")
        results.append((5, "Hyperparameter Tuning with Optuna", "FAILED", str(e)))
        failed += 1

# =============================================================================
# BLOCK 6: Neural Architecture Search
# =============================================================================
if test_block(6, "Neural Architecture Search"):
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.datasets import load_digits
        from sklearn.model_selection import cross_val_score

        # Load Digits dataset
        digits = load_digits()
        X_digits, y_digits = digits.data, digits.target

        print("Digits dataset:", X_digits.shape)
        print("Number of classes:", len(np.unique(y_digits)))

        # Define architecture search space
        def objective_architecture(trial):
            n_layers = trial.suggest_int('n_layers', 1, 3)
            hidden_layer_sizes = []
            for i in range(n_layers):
                n_units = trial.suggest_categorical(f'n_units_layer_{i}', [16, 32, 64, 128])
                hidden_layer_sizes.append(n_units)

            activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
            learning_rate = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)

            model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_layer_sizes),
                activation=activation,
                learning_rate_init=learning_rate,
                max_iter=100,
                random_state=42
            )

            scores = cross_val_score(model, X_digits, y_digits, cv=3, scoring='accuracy')
            return scores.mean()

        # Run NAS (reduced to 15 trials for speed)
        print("\n--- Neural Architecture Search with Bayesian Optimization ---")
        study_nas = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study_nas.optimize(objective_architecture, n_trials=15, show_progress_bar=False)

        print(f"\nBest accuracy: {study_nas.best_value:.4f}")
        print(f"\nBest architecture:")
        n_layers_best = study_nas.best_params['n_layers']
        print(f"  Number of layers: {n_layers_best}")
        architecture = [study_nas.best_params[f'n_units_layer_{i}']
                        for i in range(n_layers_best)]
        print(f"  Architecture: {architecture}")
        print(f"  Activation: {study_nas.best_params['activation']}")
        print(f"  Learning rate: {study_nas.best_params['learning_rate_init']:.6f}")

        # Compare with baseline
        print("\n--- Baseline: Simple hand-tuned architecture ---")
        baseline_model = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation='relu',
            learning_rate_init=0.001,
            max_iter=100,
            random_state=42
        )
        baseline_scores = cross_val_score(baseline_model, X_digits, y_digits,
                                         cv=3, scoring='accuracy')
        print(f"Baseline accuracy: {baseline_scores.mean():.4f}")
        improvement = (study_nas.best_value - baseline_scores.mean())*100
        print(f"NAS difference: {improvement:.2f}%")

        assert study_nas.best_value > 0.8, "Should achieve good accuracy on digits"

        results.append((6, "Neural Architecture Search", "PASSED", "All checks passed"))
        passed += 1
        print("✓ Block 6 PASSED")
    except Exception as e:
        print(f"✗ Block 6 FAILED: {e}")
        results.append((6, "Neural Architecture Search", "FAILED", str(e)))
        failed += 1

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("CODE REVIEW SUMMARY")
print("="*80)
print(f"\nTotal blocks tested: {total_blocks}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"\nSuccess rate: {passed/total_blocks*100:.1f}%")

print("\n" + "="*80)
print("DETAILED RESULTS")
print("="*80)
for block_num, description, status, message in results:
    print(f"\nBlock {block_num}: {description}")
    print(f"  Status: {status}")
    if status == "FAILED":
        print(f"  Error: {message}")

# Final rating
if failed == 0:
    rating = "ALL_PASS"
elif failed <= 2:
    rating = "MINOR_FIXES"
else:
    rating = "BROKEN"

print("\n" + "="*80)
print(f"FINAL RATING: {rating}")
print("="*80)
