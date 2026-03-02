#!/usr/bin/env python3
"""
Code Review Test Script for Chapter 40: Advanced Generative Models
Tests all code blocks to verify they execute correctly
"""

import sys
import traceback

# Track all issues
issues = []
blocks_tested = 0
blocks_passing = 0

def test_block(name, code_func):
    """Test a code block and report results."""
    global blocks_tested, blocks_passing, issues
    blocks_tested += 1
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        code_func()
        blocks_passing += 1
        print(f"✓ {name} PASSED")
        return True
    except Exception as e:
        error_msg = f"Block {blocks_tested}: {name}\nError: {str(e)}\n{traceback.format_exc()}"
        issues.append(error_msg)
        print(f"✗ {name} FAILED")
        print(f"Error: {e}")
        return False

# ============================================================================
# Part 1: Forward Diffusion Process
# ============================================================================
def test_part1():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from matplotlib.gridspec import GridSpec

    # Load MNIST-like digits dataset
    digits = load_digits()
    images = digits.images / 16.0  # Normalize to [0, 1]
    sample_image = images[0]  # Shape: (8, 8)

    # Linear variance schedule
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02
    beta_t = np.linspace(beta_start, beta_end, T)

    # Precompute alpha values
    alpha_t = 1.0 - beta_t
    alpha_bar_t = np.cumprod(alpha_t)

    # Set random seed for reproducibility
    np.random.seed(42)

    def forward_diffusion(x_0, t, alpha_bar):
        """
        Apply forward diffusion to timestep t.
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        noise = np.random.randn(*x_0.shape)
        sqrt_alpha_bar = np.sqrt(alpha_bar[t])
        sqrt_one_minus_alpha_bar = np.sqrt(1.0 - alpha_bar[t])
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    # Visualize forward diffusion at different timesteps
    timesteps = [0, 100, 250, 500, 750, 999]
    fig = plt.figure(figsize=(15, 3))
    gs = GridSpec(1, len(timesteps), figure=fig)

    for i, t in enumerate(timesteps):
        ax = fig.add_subplot(gs[0, i])
        if t == 0:
            noisy_image = sample_image
        else:
            noisy_image, _ = forward_diffusion(sample_image, t-1, alpha_bar_t)

        ax.imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f't = {t}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-14/ch40/forward_diffusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Forward diffusion visualization saved.")

    # Verify final timestep is approximately Gaussian noise
    final_noisy, _ = forward_diffusion(sample_image, T-1, alpha_bar_t)
    print(f"\nOriginal image: mean={sample_image.mean():.3f}, std={sample_image.std():.3f}")
    print(f"Final noisy (t={T-1}): mean={final_noisy.mean():.3f}, std={final_noisy.std():.3f}")
    print(f"Expected for Gaussian: mean≈0, std≈1")

# ============================================================================
# Part 2: Simple U-Net for Denoising
# ============================================================================
def test_part2():
    import torch
    import torch.nn as nn

    class SimpleUNet(nn.Module):
        """
        Simplified U-Net for MNIST digit denoising.
        Predicts the noise epsilon added to an image.
        """
        def __init__(self, time_dim=32):
            super().__init__()

            # Time embedding MLP
            self.time_mlp = nn.Sequential(
                nn.Linear(1, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )

            # Encoder (downsampling)
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2)

            # Bottleneck
            self.bottleneck = nn.Conv2d(64 + time_dim, 64, 3, padding=1)

            # Decoder (upsampling)
            self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.conv3 = nn.Conv2d(64, 32, 3, padding=1)  # 64 due to skip connection
            self.conv4 = nn.Conv2d(32, 1, 3, padding=1)

            self.relu = nn.ReLU()

        def forward(self, x, t):
            """
            Args:
                x: Noisy image (batch, 1, 8, 8)
                t: Timestep (batch, 1)
            Returns:
                Predicted noise (batch, 1, 8, 8)
            """
            batch_size = x.shape[0]

            # Embed timestep
            t_emb = self.time_mlp(t)  # (batch, time_dim)

            # Encoder
            x1 = self.relu(self.conv1(x))  # (batch, 32, 8, 8)
            x2 = self.pool(x1)              # (batch, 32, 4, 4)
            x2 = self.relu(self.conv2(x2)) # (batch, 64, 4, 4)

            # Inject time embedding
            t_emb_spatial = t_emb.view(batch_size, -1, 1, 1).expand(-1, -1, 4, 4)
            x2 = torch.cat([x2, t_emb_spatial], dim=1)  # (batch, 64+time_dim, 4, 4)

            # Bottleneck
            x3 = self.relu(self.bottleneck(x2))  # (batch, 64, 4, 4)

            # Decoder with skip connection
            x4 = self.upconv1(x3)  # (batch, 32, 8, 8)
            x4 = torch.cat([x4, x1], dim=1)  # (batch, 64, 8, 8) - skip connection
            x4 = self.relu(self.conv3(x4))   # (batch, 32, 8, 8)

            # Output: predicted noise
            noise_pred = self.conv4(x4)  # (batch, 1, 8, 8)

            return noise_pred

    # Test the architecture
    model = SimpleUNet(time_dim=32)
    test_input = torch.randn(4, 1, 8, 8)  # Batch of 4 noisy images
    test_time = torch.randint(0, 1000, (4, 1), dtype=torch.float32)

    with torch.no_grad():
        output = model(test_input, test_time)

    print(f"Input shape: {test_input.shape}")
    print(f"Time shape: {test_time.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Architecture has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Store model for later tests
    global global_unet
    global_unet = SimpleUNet

# ============================================================================
# Part 3: Training a DDPM
# ============================================================================
def test_part3():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.datasets import load_digits
    import numpy as np
    import matplotlib.pyplot as plt

    # Prepare data
    digits = load_digits()
    X = digits.images / 16.0  # Normalize to [0, 1]
    X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension: (N, 1, 8, 8)

    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = global_unet(time_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Training parameters
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02
    beta_t = torch.linspace(beta_start, beta_end, T, device=device)
    alpha_t = 1.0 - beta_t
    alpha_bar_t = torch.cumprod(alpha_t, dim=0)

    def get_noisy_image(x_0, t, alpha_bar, device):
        """
        Add noise to x_0 according to timestep t.
        Returns: x_t and the noise epsilon.
        """
        batch_size = x_0.shape[0]
        epsilon = torch.randn_like(x_0, device=device)

        # Gather alpha_bar values for each timestep in the batch
        sqrt_alpha_bar = torch.sqrt(alpha_bar[t]).view(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar[t]).view(batch_size, 1, 1, 1)

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon
        return x_t, epsilon

    # Training loop
    num_epochs = 20
    losses = []

    print("Training DDPM...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (x_0,) in enumerate(dataloader):
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]

            # Sample random timesteps for each image in batch
            t = torch.randint(0, T, (batch_size,), device=device)

            # Get noisy images and true noise
            x_t, epsilon_true = get_noisy_image(x_0, t, alpha_bar_t, device)

            # Predict noise
            t_input = t.view(-1, 1).float()
            epsilon_pred = model(x_t, t_input)

            # Compute loss: MSE between predicted and true noise
            loss = criterion(epsilon_pred, epsilon_true)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training complete.")

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DDPM Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/chirag/ds-book/book/course-14/ch40/ddpm_training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Loss curve saved.")

    # Store for later tests
    global global_model, global_device, global_T, global_alpha_t, global_alpha_bar_t, global_beta_t
    global_model = model
    global_device = device
    global_T = T
    global_alpha_t = alpha_t
    global_alpha_bar_t = alpha_bar_t
    global_beta_t = beta_t

# ============================================================================
# Part 4: DDPM Sampling
# ============================================================================
def test_part4():
    import torch
    import matplotlib.pyplot as plt
    import time

    @torch.no_grad()
    def ddpm_sample(model, shape, T, alpha_t, alpha_bar_t, beta_t, device):
        """
        Sample from the model using DDPM (Markovian sampling).
        Requires T steps (slow but highest quality).
        """
        model.eval()

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        # Reverse diffusion: iterate from T-1 to 0
        for t_idx in reversed(range(T)):
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            t_input = t.view(-1, 1).float()

            # Predict noise
            epsilon_pred = model(x_t, t_input)

            # Compute denoising step
            alpha_t_val = alpha_t[t_idx]
            alpha_bar_t_val = alpha_bar_t[t_idx]
            beta_t_val = beta_t[t_idx]

            # Mean of reverse distribution
            coeff1 = 1.0 / torch.sqrt(alpha_t_val)
            coeff2 = beta_t_val / torch.sqrt(1.0 - alpha_bar_t_val)
            mean = coeff1 * (x_t - coeff2 * epsilon_pred)

            if t_idx > 0:
                # Add noise (stochastic sampling)
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(beta_t_val)
                x_t = mean + sigma_t * noise
            else:
                # Final step: no noise
                x_t = mean

        return x_t

    # Generate samples
    num_samples = 8
    sample_shape = (num_samples, 1, 8, 8)

    print("Generating samples with DDPM (1000 steps)...")
    start_time = time.time()
    generated_images = ddpm_sample(global_model, sample_shape, global_T, global_alpha_t, global_alpha_bar_t, global_beta_t, global_device)
    ddpm_time = time.time() - start_time
    print(f"DDPM sampling took {ddpm_time:.2f} seconds")

    # Visualize generated samples
    generated_images = generated_images.cpu().numpy()
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 2))
    for i in range(num_samples):
        axes[i].imshow(generated_images[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
    plt.suptitle('DDPM Generated Digits')
    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-14/ch40/ddpm_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated samples saved.")

    global global_ddpm_time, global_ddpm_sample_func
    global_ddpm_time = ddpm_time
    global_ddpm_sample_func = ddpm_sample

# ============================================================================
# Part 5: DDIM Sampling
# ============================================================================
def test_part5():
    import torch
    import matplotlib.pyplot as plt
    import time

    @torch.no_grad()
    def ddim_sample(model, shape, num_steps, T, alpha_bar_t, device):
        """
        Sample using DDIM (non-Markovian, deterministic sampling).
        Can use far fewer steps (e.g., 50 instead of 1000).
        """
        model.eval()

        # Create subsequence of timesteps
        step_size = T // num_steps
        timesteps = list(range(0, T, step_size))[::-1]  # Reverse order

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        for i, t_idx in enumerate(timesteps):
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            t_input = t.view(-1, 1).float()

            # Predict noise
            epsilon_pred = model(x_t, t_input)

            # Get alpha values
            alpha_bar_t_val = alpha_bar_t[t_idx]

            # Predict x_0 from x_t
            pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t_val) * epsilon_pred) / torch.sqrt(alpha_bar_t_val)
            pred_x0 = torch.clamp(pred_x0, 0, 1)  # Clip to valid range

            if i < len(timesteps) - 1:
                # Get next timestep
                t_next_idx = timesteps[i + 1]
                alpha_bar_t_next = alpha_bar_t[t_next_idx]

                # DDIM deterministic update
                x_t = (torch.sqrt(alpha_bar_t_next) * pred_x0 +
                       torch.sqrt(1.0 - alpha_bar_t_next) * epsilon_pred)
            else:
                x_t = pred_x0

        return x_t

    # Generate samples with DDIM
    num_samples = 8
    sample_shape = (num_samples, 1, 8, 8)
    num_steps_ddim = 50

    print(f"\nGenerating samples with DDIM ({num_steps_ddim} steps)...")
    start_time = time.time()
    generated_images_ddim = ddim_sample(global_model, sample_shape, num_steps_ddim, global_T, global_alpha_bar_t, global_device)
    ddim_time = time.time() - start_time
    print(f"DDIM sampling took {ddim_time:.2f} seconds")
    print(f"Speedup: {global_ddpm_time / ddim_time:.1f}x faster than DDPM")

    # Visualize DDIM samples
    generated_images_ddim = generated_images_ddim.cpu().numpy()
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 2))
    for i in range(num_samples):
        axes[i].imshow(generated_images_ddim[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
    plt.suptitle(f'DDIM Generated Digits ({num_steps_ddim} steps)')
    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-14/ch40/ddim_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("DDIM samples saved.")

    global global_ddim_sample_func
    global_ddim_sample_func = ddim_sample

# ============================================================================
# Part 6: Classifier-Free Guidance
# ============================================================================
def test_part6():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.datasets import load_digits

    class ConditionalUNet(nn.Module):
        """
        U-Net with class conditioning for classifier-free guidance.
        """
        def __init__(self, num_classes=10, time_dim=32, class_emb_dim=16):
            super().__init__()

            # Time embedding
            self.time_mlp = nn.Sequential(
                nn.Linear(1, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )

            # Class embedding (with special null token for unconditional)
            self.class_embedding = nn.Embedding(num_classes + 1, class_emb_dim)

            # Encoder
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2)

            # Bottleneck (with time and class conditioning)
            self.bottleneck = nn.Conv2d(64 + time_dim + class_emb_dim, 64, 3, padding=1)

            # Decoder
            self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
            self.conv4 = nn.Conv2d(32, 1, 3, padding=1)

            self.relu = nn.ReLU()
            self.num_classes = num_classes

        def forward(self, x, t, labels):
            """
            Args:
                x: Noisy image (batch, 1, 8, 8)
                t: Timestep (batch, 1)
                labels: Class labels (batch,), use num_classes for null token
            """
            batch_size = x.shape[0]

            # Embed time and class
            t_emb = self.time_mlp(t)  # (batch, time_dim)
            c_emb = self.class_embedding(labels)  # (batch, class_emb_dim)

            # Encoder
            x1 = self.relu(self.conv1(x))
            x2 = self.pool(x1)
            x2 = self.relu(self.conv2(x2))

            # Inject time and class embeddings
            t_emb_spatial = t_emb.view(batch_size, -1, 1, 1).expand(-1, -1, 4, 4)
            c_emb_spatial = c_emb.view(batch_size, -1, 1, 1).expand(-1, -1, 4, 4)
            x2 = torch.cat([x2, t_emb_spatial, c_emb_spatial], dim=1)

            # Bottleneck
            x3 = self.relu(self.bottleneck(x2))

            # Decoder
            x4 = self.upconv1(x3)
            x4 = torch.cat([x4, x1], dim=1)
            x4 = self.relu(self.conv3(x4))
            noise_pred = self.conv4(x4)

            return noise_pred

    # Train conditional model
    torch.manual_seed(42)
    cond_model = ConditionalUNet(num_classes=10, time_dim=32).to(global_device)
    cond_optimizer = optim.Adam(cond_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Prepare labeled dataset
    digits = load_digits()
    X = torch.FloatTensor(digits.images / 16.0).unsqueeze(1)
    y = torch.LongTensor(digits.target)
    cond_dataset = TensorDataset(X, y)
    cond_dataloader = DataLoader(cond_dataset, batch_size=32, shuffle=True)

    def get_noisy_image(x_0, t, alpha_bar, device):
        batch_size = x_0.shape[0]
        epsilon = torch.randn_like(x_0, device=device)
        sqrt_alpha_bar = torch.sqrt(alpha_bar[t]).view(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar[t]).view(batch_size, 1, 1, 1)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon
        return x_t, epsilon

    print("\nTraining conditional DDPM with classifier-free guidance...")
    num_epochs_cond = 15
    null_token = 10  # Index for unconditional
    p_uncond = 0.1  # Probability of dropping condition

    for epoch in range(num_epochs_cond):
        epoch_loss = 0
        for batch_idx, (x_0, labels) in enumerate(cond_dataloader):
            x_0 = x_0.to(global_device)
            labels = labels.to(global_device)
            batch_size = x_0.shape[0]

            # Randomly drop labels for unconditional training
            uncond_mask = torch.rand(batch_size, device=global_device) < p_uncond
            labels_train = labels.clone()
            labels_train[uncond_mask] = null_token

            # Sample timesteps and add noise
            t = torch.randint(0, global_T, (batch_size,), device=global_device)
            x_t, epsilon_true = get_noisy_image(x_0, t, global_alpha_bar_t, global_device)

            # Predict noise
            t_input = t.view(-1, 1).float()
            epsilon_pred = cond_model(x_t, t_input, labels_train)

            # Compute loss
            loss = criterion(epsilon_pred, epsilon_true)

            cond_optimizer.zero_grad()
            loss.backward()
            cond_optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(cond_dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs_cond}], Loss: {avg_loss:.4f}")

    print("Conditional training complete.")

    global global_cond_model
    global_cond_model = cond_model

# ============================================================================
# Part 7: Sampling with Guidance
# ============================================================================
def test_part7():
    import torch
    import matplotlib.pyplot as plt

    @torch.no_grad()
    def guided_ddim_sample(model, shape, target_class, guidance_scale, num_steps, T, alpha_bar_t, device, null_token=10):
        """
        Sample with classifier-free guidance.
        epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)
        """
        model.eval()

        # Create subsequence of timesteps
        step_size = T // num_steps
        timesteps = list(range(0, T, step_size))[::-1]

        # Start from noise
        x_t = torch.randn(shape, device=device)

        # Prepare labels
        batch_size = shape[0]
        cond_labels = torch.full((batch_size,), target_class, device=device, dtype=torch.long)
        uncond_labels = torch.full((batch_size,), null_token, device=device, dtype=torch.long)

        for i, t_idx in enumerate(timesteps):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            t_input = t.view(-1, 1).float()

            # Predict noise: both conditional and unconditional
            epsilon_cond = model(x_t, t_input, cond_labels)
            epsilon_uncond = model(x_t, t_input, uncond_labels)

            # Apply classifier-free guidance
            epsilon_pred = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)

            # DDIM update
            alpha_bar_t_val = alpha_bar_t[t_idx]
            pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t_val) * epsilon_pred) / torch.sqrt(alpha_bar_t_val)
            pred_x0 = torch.clamp(pred_x0, 0, 1)

            if i < len(timesteps) - 1:
                t_next_idx = timesteps[i + 1]
                alpha_bar_t_next = alpha_bar_t[t_next_idx]
                x_t = (torch.sqrt(alpha_bar_t_next) * pred_x0 +
                       torch.sqrt(1.0 - alpha_bar_t_next) * epsilon_pred)
            else:
                x_t = pred_x0

        return x_t

    # Generate digit "7" with different guidance scales
    target_digit = 7
    guidance_scales = [0.0, 1.0, 3.0, 5.0]
    sample_shape = (4, 1, 8, 8)

    print("\nGenerating digit '7' with different guidance scales...")
    fig, axes = plt.subplots(len(guidance_scales), 4, figsize=(8, 8))

    for row, w in enumerate(guidance_scales):
        samples = guided_ddim_sample(
            global_cond_model, sample_shape, target_digit, w, 50, global_T, global_alpha_bar_t, global_device
        )
        samples = samples.cpu().numpy()

        for col in range(4):
            axes[row, col].imshow(samples[col, 0], cmap='gray', vmin=0, vmax=1)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(f'w={w}', fontsize=12)

    plt.suptitle("Classifier-Free Guidance: Generating Digit '7'")
    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-14/ch40/guidance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Guidance comparison saved.")

# ============================================================================
# Part 8: Stable Diffusion Architecture (High-Level)
# ============================================================================
def test_part8():
    # This is just a text-based demonstration, not executable code
    print("\nStable Diffusion Architecture:")
    print("=" * 60)
    print("Input: Text prompt")
    print("  ↓ [CLIP Tokenizer]")
    print("77 tokens")
    print("  ↓ [CLIP Text Encoder - frozen]")
    print("Text embeddings: 77 × 768")
    print("  ↓ (used in cross-attention)")
    print("")
    print("Latent: Random noise (4 × 64 × 64)")
    print("  ↓ [U-Net with Cross-Attention - learned]")
    print("  | - Self-attention layers")
    print("  | - Cross-attention with text embeddings")
    print("  | - Iterative denoising (50-100 steps)")
    print("  ↓")
    print("Denoised latent: (4 × 64 × 64)")
    print("  ↓ [VAE Decoder - learned]")
    print("RGB Image: (3 × 512 × 512)")
    print("=" * 60)
    print("\nKey Innovation: Diffusion in latent space (not pixel space)")
    print("Compression ratio: 512×512×3 / (64×64×4) ≈ 48× reduction")
    print("This enables high-resolution generation on consumer GPUs.")

# ============================================================================
# Run all tests
# ============================================================================
if __name__ == "__main__":
    print("Starting Code Review for Chapter 40: Advanced Generative Models")
    print("="*70)

    # Initialize global variables
    global_unet = None
    global_model = None
    global_device = None
    global_T = None
    global_alpha_t = None
    global_alpha_bar_t = None
    global_beta_t = None
    global_ddpm_time = None
    global_ddpm_sample_func = None
    global_ddim_sample_func = None
    global_cond_model = None

    # Test each part
    test_block("Part 1: Forward Diffusion Process", test_part1)
    test_block("Part 2: Simple U-Net for Denoising", test_part2)
    test_block("Part 3: Training a DDPM", test_part3)
    test_block("Part 4: DDPM Sampling", test_part4)
    test_block("Part 5: DDIM Sampling", test_part5)
    test_block("Part 6: Classifier-Free Guidance", test_part6)
    test_block("Part 7: Sampling with Guidance", test_part7)
    test_block("Part 8: Stable Diffusion Architecture", test_part8)

    # Summary
    print("\n" + "="*70)
    print("TESTING SUMMARY")
    print("="*70)
    print(f"Total blocks tested: {blocks_tested}")
    print(f"Passing: {blocks_passing}")
    print(f"Failing: {blocks_tested - blocks_passing}")

    if issues:
        print("\n" + "="*70)
        print("ISSUES FOUND:")
        print("="*70)
        for issue in issues:
            print(issue)
            print("-"*70)
    else:
        print("\n✓ All code blocks passed!")

    sys.exit(0 if len(issues) == 0 else 1)
