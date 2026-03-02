# Chapter 40 Diagrams

This directory contains all visualizations for Chapter 40: Advanced Generative Models.

## Generated Diagrams

### 1. **diffusion_process.md** (Mermaid)
   - Conceptual diagram showing forward and reverse diffusion processes
   - Illustrates the symmetric nature of noise addition and removal
   - Color-coded to show progression from clean data to noise and back

### 2. **forward_diffusion.png** (Matplotlib)
   - Visualization of forward diffusion at timesteps [0, 100, 250, 500, 750, 999]
   - Shows gradual noise addition to a digit image
   - Demonstrates how structure degrades over time

### 3. **ddpm_training_loss.png** (Matplotlib)
   - Training loss curve for DDPM model
   - Shows convergence over 20 epochs
   - Includes annotations for initial and final loss values

### 4. **ddpm_samples.png** (Matplotlib)
   - 8 sample digits generated using DDPM (1000 steps)
   - Demonstrates high-quality generation with full Markovian sampling

### 5. **ddim_samples.png** (Matplotlib)
   - 8 sample digits generated using DDIM (50 steps)
   - Shows comparable quality to DDPM with 20× speedup

### 6. **guidance_comparison.png** (Matplotlib)
   - Classifier-free guidance comparison for digit "7"
   - Shows effect of guidance scales w ∈ [0, 1, 3, 5]
   - 4 samples per guidance scale

### 7. **stable_diffusion_arch.png** (Matplotlib)
   - Detailed visual architecture diagram of Stable Diffusion
   - Shows flow from text prompt through CLIP encoder, U-Net, and VAE decoder
   - Highlights key innovation: latent diffusion with 48× compression

### 8. **stable_diffusion_arch.md** (Mermaid)
   - Flowchart version of Stable Diffusion architecture
   - Companion diagram to the visual version

### 9. **variance_schedules.png** (Matplotlib)
   - Comparison of linear vs cosine variance schedules
   - Two subplots: β_t values and cumulative α̅_t
   - Shows how cosine schedule preserves more signal early on

## Color Palette

All diagrams use a consistent color scheme:
- Blue (#2196F3): Primary/initial states
- Green (#4CAF50): Final/success states  
- Orange (#FF9800): Intermediate/noise states
- Purple (#9C27B0): Processing/model states
- Red (#F44336): Output/attention states
- Gray (#607D8B): Auxiliary/annotation elements

## Usage in Content

All diagrams are referenced in `content.md` with appropriate context and explanations. The visualizations support:
- Part 1: Forward Diffusion Process
- Part 3: DDPM Training
- Part 4: DDPM Sampling
- Part 5: DDIM Sampling
- Part 7: Classifier-Free Guidance
- Part 8: Stable Diffusion Architecture
- Solution 2: Variance Schedules

## Technical Notes

- All PNG images are saved at 150 DPI with white backgrounds
- Figures use tight_layout() for optimal spacing
- Font sizes are kept at 12pt minimum for readability
- All images are under 800px width for textbook compatibility
