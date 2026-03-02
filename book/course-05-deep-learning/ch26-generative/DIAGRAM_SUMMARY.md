# Chapter 26: Generative Models - Diagram Summary

## Generated Diagrams

All diagrams for Chapter 26 have been successfully generated and saved to the `diagrams/` directory.

### Main Content Diagrams

1. **discriminative_vs_generative.png** (192 KB)
   - Location: Line 172
   - Type: Matplotlib comparison visualization
   - Shows: Side-by-side comparison of discriminative vs generative models
   - Colors: Blue (#2196F3) and Red (#F44336) for classes

2. **vae_training_curves.png** (81 KB)
   - Location: Line 363
   - Type: Matplotlib line plots
   - Shows: VAE training progress with total loss and loss components (reconstruction + KL divergence)

3. **vae_reconstruction_generation.png** (62 KB)
   - Location: Line 425
   - Type: Matplotlib image grid
   - Shows: 3 rows × 10 columns showing original, reconstructed, and generated MNIST digits

4. **vae_interpolation.png** (26 KB)
   - Location: Line 484
   - Type: Matplotlib image sequence
   - Shows: Latent space interpolation morphing digit "3" to digit "8" with 10 intermediate steps

5. **gan_training_curves.png** (74 KB)
   - Location: Line 676
   - Type: Matplotlib line plots
   - Shows: GAN adversarial training curves for generator and discriminator losses

6. **gan_generated_samples.png** (79 KB)
   - Location: Line 711
   - Type: Matplotlib image grid
   - Shows: 5×5 grid of GAN-generated MNIST digits

7. **cgan_controlled_generation.png** (192 KB)
   - Location: Line 875
   - Type: Matplotlib image grid
   - Shows: 10 rows × 10 columns, each row showing different samples of the same digit (0-9)

8. **diffusion_forward_process.png** (36 KB)
   - Location: Line 939
   - Type: Matplotlib image sequence
   - Shows: Forward diffusion process at t = {0, 250, 500, 750, 1000}

9. **diffusion_architecture_diagram.png** (104 KB)
   - Location: Line 1040
   - Type: Matplotlib diagram with boxes and arrows
   - Shows: Complete diffusion model pipeline with forward and reverse processes

### Exercise Solution Diagrams

10. **vae_latent_dim_comparison.png** (410 KB)
    - Exercise: Solution 1
    - Shows: Comparison of VAE generation quality across latent dimensions {2, 10, 20, 50}

11. **gan_mode_collapse_detection.png** (59 KB)
    - Exercise: Solution 2
    - Shows: Histogram of generated digit distribution to detect mode collapse

12. **vae_latent_arithmetic.png** (30 KB)
    - Exercise: Solution 4
    - Shows: Latent space arithmetic for attribute manipulation (thickness transfer)

13. **diffusion_multiple_digits.png** (109 KB)
    - Exercise: Solution 5
    - Shows: Forward diffusion process applied to 5 different digits simultaneously

## Design Specifications

### Color Palette (Consistent across all diagrams)
- Primary Blue: #2196F3 (used for discriminators, losses, main data)
- Red: #F44336 (used for generators, secondary losses, fake data)
- Orange: #FF9800 (used for generated samples, highlights)
- Green: #4CAF50 (not used in current diagrams)
- Purple: #9C27B0 (not used in current diagrams)
- Gray: #607D8B (used for neutral elements, equilibrium lines)

### Technical Specifications
- Resolution: 150 DPI
- Max width: 800px (achieved through figure size)
- Background: White
- Font sizes:
  - Titles: 14pt (bold)
  - Axis labels: 12pt
  - Annotations: 9-11pt
- Format: PNG with tight bounding box

### Visualization Techniques Used
- **Matplotlib/Seaborn**: All diagrams use matplotlib for plotting
- **Scipy**: Gaussian filters for simulated MNIST digits, KDE for density estimation
- **NumPy**: Data generation and manipulation
- **FancyBboxPatch**: Rounded boxes for architecture diagrams
- **FancyArrowPatch**: Arrows for process flows

## Notes

- All diagrams follow the textbook's educational style with clear labels and annotations
- MNIST digits are simulated using programmatic generation (not real MNIST data) to avoid dependencies
- The diagrams demonstrate concepts without requiring PyTorch/training (conceptual focus)
- All images use consistent styling and color schemes for visual cohesion
- Diagrams are optimized for both digital reading and print at 150 DPI

## Generation Script

The complete generation script is available at: `generate_diagrams.py`

To regenerate all diagrams:
```bash
cd book/course-05-deep-learning/ch26-generative
python generate_diagrams.py
```

---
Generated on: 2026-03-01
Total diagrams: 13
Total size: ~1.5 MB
