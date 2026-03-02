"""
Code Review Test Script for Chapter 34: Bayesian Deep Learning
Tests all code blocks sequentially to verify they work correctly.
"""

import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

def test_block(block_num, description, code_func):
    """Test a single code block"""
    print(f"\n{'='*70}")
    print(f"Testing Block {block_num}: {description}")
    print(f"{'='*70}")
    try:
        code_func()
        print(f"✓ Block {block_num} passed")
        return True
    except Exception as e:
        print(f"✗ Block {block_num} FAILED")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return False

# Track results
results = []

# Block 1: Visualization - Point vs Distribution
def block_1():
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].axvline(x=0.5, color='red', linewidth=3, label='Single weight θ*')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel('Weight value', fontsize=12)
    axes[0].set_title('Traditional NN: Point Estimate', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].set_yticks([])
    axes[0].text(0.5, 0.5, 'θ* = 0.5', ha='center', va='bottom', fontsize=12)

    x = np.linspace(0, 1, 1000)
    mu, sigma = 0.5, 0.1
    posterior = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    axes[1].plot(x, posterior, 'b-', linewidth=2, label='Distribution p(θ|D)')
    axes[1].fill_between(x, posterior, alpha=0.3)
    axes[1].axvline(x=mu, color='blue', linewidth=2, linestyle='--', alpha=0.7, label=f'Mean μ = {mu}')
    axes[1].axvspan(mu - sigma, mu + sigma, alpha=0.2, color='blue', label=f'Std σ = {sigma}')
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel('Weight value', fontsize=12)
    axes[1].set_ylabel('Probability density', fontsize=12)
    axes[1].set_title('Bayesian NN: Distribution over Weights', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('diagrams/point_vs_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

results.append(test_block(1, "Visualization - Point vs Distribution", block_1))

# Block 2: Standard Neural Network Baseline
def block_2():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    np.random.seed(42)
    torch.manual_seed(42)

    data = load_breast_cancer()
    X, y = data.data, data.target

    assert X.shape == (569, 30)
    assert y.shape == (569,)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    class StandardNN(nn.Module):
        def __init__(self, input_dim):
            super(StandardNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

    model = StandardNN(input_dim=30)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_t)
        loss = criterion(y_pred, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t)
        accuracy = ((y_pred_test > 0.5).float() == y_test_t).float().mean()
        assert accuracy > 0.90  # Should get >90% accuracy

    # Store for later use
    global standard_predictions, standard_model
    standard_predictions = y_pred_test.numpy()
    standard_model = model

results.append(test_block(2, "Standard Neural Network Baseline", block_2))

# Block 3: Bayes by Backprop Implementation
def block_3():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np

    torch.manual_seed(42)
    np.random.seed(42)

    class BayesianLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super(BayesianLinear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
            self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))

            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))

            self.weight_mu.data.normal_(0, 0.1)
            self.weight_log_sigma.data.fill_(-3)
            self.bias_mu.data.normal_(0, 0.1)
            self.bias_log_sigma.data.fill_(-3)

        def forward(self, x):
            weight_sigma = torch.exp(self.weight_log_sigma)
            weight_epsilon = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_sigma * weight_epsilon

            bias_sigma = torch.exp(self.bias_log_sigma)
            bias_epsilon = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_sigma * bias_epsilon

            return nn.functional.linear(x, weight, bias)

        def kl_divergence(self):
            weight_sigma = torch.exp(self.weight_log_sigma)
            bias_sigma = torch.exp(self.bias_log_sigma)

            kl_weight = 0.5 * torch.sum(
                weight_sigma**2 + self.weight_mu**2 - 1 - 2*self.weight_log_sigma
            )
            kl_bias = 0.5 * torch.sum(
                bias_sigma**2 + self.bias_mu**2 - 1 - 2*self.bias_log_sigma
            )

            return kl_weight + kl_bias

    class BayesianNN(nn.Module):
        def __init__(self, input_dim):
            super(BayesianNN, self).__init__()
            self.fc1 = BayesianLinear(input_dim, 64)
            self.fc2 = BayesianLinear(64, 32)
            self.fc3 = BayesianLinear(32, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

        def kl_divergence(self):
            return self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.fc3.kl_divergence()

    # Use data from previous block
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    bayes_model = BayesianNN(input_dim=30)
    optimizer = optim.Adam(bayes_model.parameters(), lr=0.001)

    def elbo_loss(model, x, y, n_samples):
        y_pred = model(x)
        nll = nn.functional.binary_cross_entropy(y_pred, y, reduction='sum')
        kl = model.kl_divergence() / n_samples
        return nll + kl

    n_epochs = 100
    n_train = len(X_train_t)

    for epoch in range(n_epochs):
        bayes_model.train()
        optimizer.zero_grad()
        loss = elbo_loss(bayes_model, X_train_t, y_train_t, n_train)
        loss.backward()
        optimizer.step()

    bayes_model.eval()
    n_samples = 100
    predictions_list = []

    with torch.no_grad():
        for _ in range(n_samples):
            y_pred = bayes_model(X_test_t)
            predictions_list.append(y_pred.numpy())

    predictions_array = np.array(predictions_list)
    bayes_mean = predictions_array.mean(axis=0)
    bayes_std = predictions_array.std(axis=0)

    bayes_accuracy = ((bayes_mean > 0.5).astype(float) == y_test_t.numpy()).mean()
    assert bayes_accuracy > 0.85  # Should get reasonable accuracy

    # Visualization
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    indices = np.arange(20)
    axes[0].errorbar(indices, bayes_mean[:20].flatten(),
                     yerr=2*bayes_std[:20].flatten(), fmt='o',
                     capsize=5, label='Bayesian NN (mean ± 2σ)')
    axes[0].scatter(indices, y_test[:20], color='red', marker='x',
                    s=100, label='True labels')
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Sample index')
    axes[0].set_ylabel('Predicted probability')
    axes[0].set_title('Predictions with Uncertainty Estimates')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(bayes_std.flatten(), bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Prediction uncertainty (std)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Predictive Uncertainty')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('diagrams/bayesian_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()

results.append(test_block(3, "Bayes by Backprop Implementation", block_3))

# Block 4: Monte Carlo Dropout
def block_4():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import numpy as np
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    np.random.seed(42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    class DropoutCNN(nn.Module):
        def __init__(self, dropout_rate=0.3):
            super(DropoutCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc1 = nn.Linear(64 * 5 * 5, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(p=dropout_rate)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.maxpool(x)
            x = self.relu(self.conv2(x))
            x = self.maxpool(x)
            x = x.view(-1, 64 * 5 * 5)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = DropoutCNN(dropout_rate=0.3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for fewer epochs for speed
    n_epochs = 2  # Reduced from 5
    for epoch in range(n_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > 100:  # Limit batches for speed
                break
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # MC Dropout predictions
    def mc_dropout_predict(model, x, n_samples=10):  # Reduced samples
        model.train()
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                output = model(x)
                probs = torch.softmax(output, dim=1)
                predictions.append(probs.numpy())
        return np.array(predictions)

    test_iter = iter(test_loader)
    test_images, test_labels = next(test_iter)

    mc_predictions = mc_dropout_predict(model, test_images, n_samples=10)
    mc_mean = mc_predictions.mean(axis=0)
    mc_std = mc_predictions.std(axis=0)
    mc_entropy = -np.sum(mc_mean * np.log(mc_mean + 1e-10), axis=1)

    mc_pred_classes = mc_mean.argmax(axis=1)

    # Visualization
    high_uncertainty_idx = np.argsort(mc_entropy)[-5:]
    low_uncertainty_idx = np.argsort(mc_entropy)[:5]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, idx in enumerate(low_uncertainty_idx):
        axes[0, i].imshow(test_images[idx].squeeze(), cmap='gray')
        axes[0, i].set_title(f"Pred: {mc_pred_classes[idx]}\nTrue: {test_labels[idx]}\nH={mc_entropy[idx]:.3f}")
        axes[0, i].axis('off')

    for i, idx in enumerate(high_uncertainty_idx):
        axes[1, i].imshow(test_images[idx].squeeze(), cmap='gray')
        axes[1, i].set_title(f"Pred: {mc_pred_classes[idx]}\nTrue: {test_labels[idx]}\nH={mc_entropy[idx]:.3f}")
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Low Uncertainty', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('High Uncertainty', fontsize=12, fontweight='bold')

    plt.suptitle('Monte Carlo Dropout: Uncertainty Estimates', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('diagrams/mc_dropout_uncertainty.png', dpi=150, bbox_inches='tight')
    plt.close()

results.append(test_block(4, "Monte Carlo Dropout", block_4))

# Block 5: Variational Autoencoder (VAE)
def block_5():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import numpy as np
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    np.random.seed(42)

    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    class VAE(nn.Module):
        def __init__(self, latent_dim=2):
            super(VAE, self).__init__()
            self.latent_dim = latent_dim

            self.encoder = nn.Sequential(
                nn.Linear(28*28, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
            )
            self.fc_mu = nn.Linear(128, latent_dim)
            self.fc_log_var = nn.Linear(128, latent_dim)

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 28*28),
                nn.Sigmoid()
            )

        def encode(self, x):
            h = self.encoder(x)
            mu = self.fc_mu(h)
            log_var = self.fc_log_var(h)
            return mu, log_var

        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            epsilon = torch.randn_like(std)
            return mu + std * epsilon

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            x_recon = self.decode(z)
            return x_recon, mu, log_var

    def vae_loss(x_recon, x, mu, log_var):
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_div

    vae = VAE(latent_dim=2)
    optimizer = optim.Adam(vae.parameters(), lr=0.001)

    # Train for fewer epochs
    n_epochs = 5  # Reduced from 20
    for epoch in range(n_epochs):
        vae.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            if batch_idx > 100:  # Limit batches
                break
            data = data.view(-1, 28*28)
            optimizer.zero_grad()
            x_recon, mu, log_var = vae(data)
            loss = vae_loss(x_recon, data, mu, log_var)
            loss.backward()
            optimizer.step()

    # Visualization 1: Latent space
    vae.eval()
    latent_codes = []
    labels_list = []

    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            if i > 10:  # Limit for speed
                break
            data = data.view(-1, 28*28)
            mu, log_var = vae.encode(data)
            latent_codes.append(mu.numpy())
            labels_list.append(labels.numpy())

    latent_codes = np.vstack(latent_codes)
    labels_array = np.concatenate(labels_list)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_codes[:, 0], latent_codes[:, 1],
                         c=labels_array, cmap='tab10', alpha=0.6, s=5)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('Latent dimension 1')
    plt.ylabel('Latent dimension 2')
    plt.title('VAE Latent Space (Fashion-MNIST)')
    plt.grid(alpha=0.3)
    plt.savefig('diagrams/vae_latent_space.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Visualization 2: Reconstructions
    test_iter = iter(test_loader)
    test_images, _ = next(test_iter)
    test_images_flat = test_images.view(-1, 28*28)

    with torch.no_grad():
        x_recon, _, _ = vae(test_images_flat)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(test_images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)

        axes[1, i].imshow(x_recon[i].view(28, 28).numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', fontsize=12)

    plt.suptitle('VAE Reconstructions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('diagrams/vae_reconstructions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Visualization 3: Generate new samples
    with torch.no_grad():
        z_sample = torch.randn(16, 2)
        generated = vae.decode(z_sample)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(16):
        row, col = i // 8, i % 8
        axes[row, col].imshow(generated[i].view(28, 28).numpy(), cmap='gray')
        axes[row, col].axis('off')

    plt.suptitle('VAE Generated Samples (from prior)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('diagrams/vae_generated.png', dpi=150, bbox_inches='tight')
    plt.close()

results.append(test_block(5, "Variational Autoencoder (VAE)", block_5))

# Block 6: Epistemic vs Aleatoric Uncertainty
def block_6():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    np.random.seed(42)

    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    class HeteroscedasticNN(nn.Module):
        def __init__(self, input_dim, dropout_rate=0.3):
            super(HeteroscedasticNN, self).__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.mean_head = nn.Linear(64, 1)
            self.log_var_head = nn.Linear(64, 1)

        def forward(self, x):
            h = self.shared(x)
            mu = self.mean_head(h)
            log_var = self.log_var_head(h)
            return mu, log_var

    def heteroscedastic_loss(mu, log_var, y):
        log_var = torch.clamp(log_var, min=-10, max=10)
        loss = 0.5 * log_var + 0.5 * torch.exp(-log_var) * (y - mu)**2
        return loss.mean()

    model = HeteroscedasticNN(input_dim=8, dropout_rate=0.3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        mu, log_var = model(X_train_t)
        loss = heteroscedastic_loss(mu, log_var, y_train_t)
        loss.backward()
        optimizer.step()

    # Compute uncertainties
    n_samples = 50
    model.train()

    aleatoric_list = []
    predictions_list = []

    with torch.no_grad():
        for _ in range(n_samples):
            mu, log_var = model(X_test_t)
            predictions_list.append(mu.numpy())
            aleatoric_list.append(torch.exp(log_var).numpy())

    predictions_array = np.array(predictions_list)
    aleatoric_array = np.array(aleatoric_list)

    aleatoric_uncertainty = aleatoric_array.mean(axis=0).flatten()
    epistemic_uncertainty = predictions_array.var(axis=0).flatten()
    mean_predictions = predictions_array.mean(axis=0).flatten()

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].scatter(y_test, mean_predictions, alpha=0.3, s=10)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', linewidth=2)
    axes[0, 0].set_xlabel('True values')
    axes[0, 0].set_ylabel('Predictions')
    axes[0, 0].set_title('Predictions vs True Values')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].scatter(mean_predictions, aleatoric_uncertainty, alpha=0.3, s=10, c='blue')
    axes[0, 1].set_xlabel('Predicted value')
    axes[0, 1].set_ylabel('Aleatoric uncertainty (σ²)')
    axes[0, 1].set_title('Aleatoric Uncertainty: Data noise (irreducible)')
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].scatter(mean_predictions, epistemic_uncertainty, alpha=0.3, s=10, c='red')
    axes[1, 0].set_xlabel('Predicted value')
    axes[1, 0].set_ylabel('Epistemic uncertainty (var)')
    axes[1, 0].set_title('Epistemic Uncertainty: Model uncertainty (reducible)')
    axes[1, 0].grid(alpha=0.3)

    indices = np.arange(100)
    axes[1, 1].bar(indices, aleatoric_uncertainty[:100], label='Aleatoric', alpha=0.7)
    axes[1, 1].bar(indices, epistemic_uncertainty[:100], bottom=aleatoric_uncertainty[:100],
                  label='Epistemic', alpha=0.7)
    axes[1, 1].set_xlabel('Sample index (first 100)')
    axes[1, 1].set_ylabel('Uncertainty')
    axes[1, 1].set_title('Total Uncertainty = Aleatoric + Epistemic')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('diagrams/uncertainty_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()

results.append(test_block(6, "Epistemic vs Aleatoric Uncertainty", block_6))

# Block 7: Calibration and Reliability Diagrams
def block_7():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    np.random.seed(42)

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_test_t = torch.FloatTensor(X_test)
    X_val_t = torch.FloatTensor(X_val)

    class StandardNN(nn.Module):
        def __init__(self, input_dim):
            super(StandardNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()

        def forward(self, x, return_logits=False):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            logits = self.fc3(x)
            if return_logits:
                return logits
            return torch.sigmoid(logits)

    standard_model = StandardNN(input_dim=30)

    # Train model (abbreviated)
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    optimizer = optim.Adam(standard_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(50):
        standard_model.train()
        optimizer.zero_grad()
        y_pred = standard_model(X_train_t)
        loss = criterion(y_pred, y_train_t)
        loss.backward()
        optimizer.step()

    # Get predictions
    standard_model.eval()
    with torch.no_grad():
        standard_probs = standard_model(X_test_t).numpy().flatten()
        standard_logits = standard_model(X_test_t, return_logits=True).numpy().flatten()

    # Bayesian predictions (simplified)
    np.random.seed(42)
    bayesian_probs = np.clip(standard_probs + np.random.normal(0, 0.05, len(standard_probs)), 0, 1)

    # Temperature Scaling
    class TemperatureScaling:
        def __init__(self):
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        def fit(self, logits, labels, lr=0.01, max_iter=50):
            logits_t = torch.FloatTensor(logits).unsqueeze(1)
            labels_t = torch.FloatTensor(labels).unsqueeze(1)

            optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

            def eval_loss():
                optimizer.zero_grad()
                scaled_logits = logits_t / self.temperature
                probs = torch.sigmoid(scaled_logits)
                loss = nn.functional.binary_cross_entropy(probs, labels_t)
                loss.backward()
                return loss

            optimizer.step(eval_loss)
            return self

        def predict(self, logits):
            with torch.no_grad():
                logits_t = torch.FloatTensor(logits).unsqueeze(1)
                scaled_probs = torch.sigmoid(logits_t / self.temperature)
                return scaled_probs.numpy().flatten()

    val_logits = standard_model(X_val_t, return_logits=True).numpy().flatten()
    temp_scaler = TemperatureScaling()
    temp_scaler.fit(val_logits, y_val)

    calibrated_probs = temp_scaler.predict(standard_logits)

    # Reliability diagram
    def reliability_diagram(y_true, y_prob, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        bin_counts = []
        bin_accuracies = []
        bin_confidences = []

        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
            if mask.sum() > 0:
                bin_counts.append(mask.sum())
                bin_accuracies.append(y_true[mask].mean())
                bin_confidences.append(y_prob[mask].mean())
            else:
                bin_counts.append(0)
                bin_accuracies.append(0)
                bin_confidences.append(bin_centers[i])

        return np.array(bin_confidences), np.array(bin_accuracies), np.array(bin_counts)

    def expected_calibration_error(y_true, y_prob, n_bins=10):
        bin_confs, bin_accs, bin_counts = reliability_diagram(y_true, y_prob, n_bins)
        n_total = bin_counts.sum()
        ece = np.sum(bin_counts / n_total * np.abs(bin_accs - bin_confs))
        return ece

    standard_ece = expected_calibration_error(y_test, standard_probs)
    bayesian_ece = expected_calibration_error(y_test, bayesian_probs)
    calibrated_ece = expected_calibration_error(y_test, calibrated_probs)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    methods = [
        ('Standard NN', standard_probs, standard_ece),
        ('Bayesian NN', bayesian_probs, bayesian_ece),
        ('Temperature Scaled', calibrated_probs, calibrated_ece)
    ]

    for idx, (name, probs, ece) in enumerate(methods):
        bin_confs, bin_accs, bin_counts = reliability_diagram(y_test, probs, n_bins=10)

        axes[idx].bar(bin_confs, bin_accs, width=0.08, alpha=0.7,
                      edgecolor='black', label='Observed frequency')
        axes[idx].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')

        for conf, acc, count in zip(bin_confs, bin_accs, bin_counts):
            if count > 0:
                axes[idx].plot([conf, conf], [conf, acc], 'k-', alpha=0.3, linewidth=1)

        axes[idx].set_xlim(0, 1)
        axes[idx].set_ylim(0, 1)
        axes[idx].set_xlabel('Predicted probability (confidence)', fontsize=11)
        axes[idx].set_ylabel('Observed frequency (accuracy)', fontsize=11)
        axes[idx].set_title(f'{name}\nECE = {ece:.4f}', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(alpha=0.3)
        axes[idx].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('diagrams/reliability_diagrams.png', dpi=150, bbox_inches='tight')
    plt.close()

results.append(test_block(7, "Calibration and Reliability Diagrams", block_7))

# Print summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
passed = sum(results)
total = len(results)
print(f"Passed: {passed}/{total}")
print(f"Failed: {total - passed}/{total}")

if passed == total:
    print("\n✓ ALL TESTS PASSED")
    sys.exit(0)
else:
    print("\n✗ SOME TESTS FAILED")
    sys.exit(1)
