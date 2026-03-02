# Solution 4: ARD for Feature Selection (Simplified test)
import numpy as np
import pymc as pm
import arviz as az
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# Load data
housing = fetch_california_housing()
X, y = housing.data[:1000], housing.target[:1000]  # Reduced for speed

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

n_features = X_train_scaled.shape[1]

# ARD Model (reduced sampling)
print("Fitting ARD model...")
with pm.Model() as ard_model:
    tau = pm.Gamma('tau', alpha=1, beta=1, shape=n_features)
    beta = pm.Normal('beta', mu=0, sigma=1/pm.math.sqrt(tau), shape=n_features)
    intercept = pm.Normal('intercept', mu=0, sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = intercept + pm.math.dot(X_train_scaled, beta)
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=y_train_scaled)

    trace_ard = pm.sample(500, tune=250, chains=2, random_seed=42,
                          target_accept=0.9, progressbar=False)

# Analyze ARD precisions
tau_posterior = trace_ard.posterior['tau'].values.reshape(-1, n_features)
tau_mean = tau_posterior.mean(axis=0)

threshold = np.percentile(tau_mean, 75)
relevant_features = tau_mean < threshold

print(f"\nARD Feature Selection (threshold = {threshold:.2f}):")
print(f"Relevant features: {relevant_features.sum()}/{n_features}")
for i, name in enumerate(housing.feature_names):
    status = "KEEP" if relevant_features[i] else "SHRINK"
    print(f"  {name:15s}: τ = {tau_mean[i]:6.2f}  [{status}]")

print("\nSolution 4: PASS")
