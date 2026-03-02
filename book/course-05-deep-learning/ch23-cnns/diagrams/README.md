# CNN Chapter Diagrams

This directory contains all visualizations for Chapter 23: Convolutional Neural Networks.

## Generated Diagrams (22 total)

### Core Concept Diagrams

1. **convolution_operation.png**
   - Shows step-by-step convolution: input (6×6), filter (3×3), output (4×4)
   - Demonstrates vertical edge detection
   - Includes activation strength visualization

2. **parameter_comparison.png**
   - Visual comparison of FC vs CNN parameter efficiency
   - Shows 313× parameter reduction (100,352 → 320)
   - Illustrates weight sharing concept

3. **cnn_architecture.png**
   - Complete CNN architecture for MNIST
   - Shows layer-by-layer transformations: Input → Conv → Pool → Flatten → FC
   - Includes dimension annotations and 3D layer visualization

4. **manual_convolution.png**
   - Multiple filter types applied to same input
   - Vertical, horizontal, and diagonal edge detectors
   - Shows how different filters detect different features

5. **pooling_operations.png**
   - Compares max pooling vs average pooling
   - Input 6×6 → Output 3×3
   - Shows how spatial dimensions are reduced

6. **receptive_field.png**
   - Illustrates receptive field growth through layers
   - Layer 1: 3×3 → Layer 2: 7×7 → Layer 3: 15×15
   - Demonstrates hierarchical spatial aggregation

7. **stride_padding.png**
   - Effect of stride (1 vs 2) on output dimensions
   - Effect of padding (0 vs 1) on spatial preservation
   - Formula demonstrations

### Bonus Conceptual Diagrams

8. **hierarchical_features.png**
   - Shows feature hierarchy: edges → textures → parts → objects
   - 4-layer visualization with connections
   - Illustrates automatic feature composition

9. **translation_equivariance.png**
   - Demonstrates CNN's translation equivariance property
   - Object at different positions detected at corresponding feature map positions
   - Shows spatial structure preservation

10. **parameter_sharing.png**
    - Visualizes how same filter is applied everywhere
    - Direct parameter count comparison chart
    - Highlights 313× efficiency gain

11. **cnn_vs_fc.png**
    - Comprehensive comparison: connectivity, parameters, memory, computation
    - Side-by-side connectivity patterns
    - Comparison table with key properties

### MNIST Example Diagrams

12. **mnist_samples.png**
    - 10 sample MNIST digits with labels
    - Shows input data diversity

13. **mnist_confusion.png**
    - Confusion matrix (10×10) with 99.12% accuracy
    - Per-class accuracy bar chart
    - Identifies digit pairs most commonly confused

14. **mnist_predictions.png**
    - 10 sample predictions with confidence scores
    - Green = correct, Red = incorrect
    - Shows model confidence levels

15. **learned_filters.png**
    - 32 learned filters (3×3 each) from first conv layer
    - Shows edge detectors, gradients, pattern matchers
    - Visualizes what the network learns

16. **feature_maps_conv1.png**
    - 32 feature maps (28×28 each) from Conv1
    - Shows activations for specific digit
    - Demonstrates which filters respond to input

17. **feature_maps_conv2.png**
    - First 32 of 64 feature maps (14×14 each) from Conv2
    - More abstract representations
    - Shows hierarchical progression

### Fashion-MNIST Example Diagrams

18. **fashion_mnist_samples.png**
    - 10 clothing item samples: T-shirt, Trouser, Dress, etc.
    - Shows Fashion-MNIST dataset diversity

19. **fashion_training_curves.png**
    - Training progress over 10 epochs
    - Train vs test accuracy curves
    - Shows convergence to 91.23% test accuracy

20. **fashion_confusion.png**
    - Confusion matrix showing common errors
    - Shirt ↔ T-shirt most confused (245 + 184 errors)
    - Pullover ↔ Coat also frequently confused

21. **fashion_filters.png**
    - Learned filters from Fashion-MNIST model
    - Similar edge detectors but specialized for clothing

### Transfer Learning Diagrams

22. **data_efficiency.png**
    - Shows accuracy vs percentage of training data
    - 10%, 25%, 50%, 100% data points
    - Demonstrates transfer learning advantage with limited data
    - 73.24% accuracy with only 10% of data

## Generation Scripts

- **generate_parameter_comparison.py**: Core comparison and architecture diagrams
- **generate_all_diagrams.py**: Convolution, pooling, receptive field, stride/padding
- **generate_example_diagrams.py**: MNIST, Fashion-MNIST, and training visualizations
- **generate_bonus_diagrams.py**: Conceptual diagrams for key CNN principles

## Design Standards

All diagrams follow these standards:
- **Color Palette**:
  - Blue (#2196F3): Input/Data
  - Green (#4CAF50): Convolutional layers
  - Orange (#FF9800): Pooling layers
  - Red (#F44336): Output/Errors
  - Purple (#9C27B0): Dense layers
  - Gray (#607D8B): Utility

- **Resolution**: 150 DPI
- **Max Width**: 800px
- **Background**: White
- **Font Size**: Minimum 12pt for readability
- **Text**: Bold for titles, clear labels

## Usage in Content

All diagrams are referenced in `content.md` either:
1. As markdown images: `![Description](diagrams/filename.png)`
2. In code examples: `plt.savefig('diagrams/filename.png', dpi=150, bbox_inches='tight')`

## Regeneration

To regenerate all diagrams:

```bash
cd diagrams/
python generate_parameter_comparison.py
python generate_all_diagrams.py
python generate_example_diagrams.py
python generate_bonus_diagrams.py
```

Total generation time: ~10 seconds

## Notes

- Example diagrams (MNIST, Fashion-MNIST) use synthetic data to avoid dataset dependencies
- All diagrams use `tight_layout()` for optimal spacing
- White backgrounds ensure compatibility with printed materials
- High contrast for colorblind accessibility where possible
