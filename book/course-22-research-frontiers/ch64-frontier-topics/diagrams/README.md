# Chapter 64 Diagrams

This directory contains three generated diagrams for Chapter 64: Frontier Topics (2025-2026).

## Generated Diagrams

### 1. frontier_overview.png
- **Location in content.md**: After line 131 (after the code block ending with plt.savefig)
- **Description**: Two-panel figure showing:
  - Left: Test-time compute scaling curves demonstrating how different strategies (Greedy, Best-of-N, Self-Consistency, Tree Search, o1-Style) improve accuracy with increased inference compute
  - Right: Efficiency-accuracy Pareto frontier comparing quantization methods (FP16, GPTQ, AWQ, GGUF, Marlin)
- **Suggested markdown insertion**:
  ```markdown
  ![Frontier Overview: Test-Time Compute Scaling and Efficiency-Accuracy Trade-off](diagrams/frontier_overview.png)
  ```

### 2. induction_head_attention.png
- **Location in content.md**: After line 788 (after plt.savefig call)
- **Description**: Attention heatmap visualizing induction head behavior in a toy transformer. Shows how the model attends to tokens that came after previous occurrences of the current token, enabling in-context learning.
- **Example sequence**: ["The", "cat", "sat", "on", "the", "mat", "and", "the", "cat", "ran"]
- **Suggested markdown insertion**:
  ```markdown
  ![Induction Head Attention Pattern](diagrams/induction_head_attention.png)
  ```

### 3. quantization_comparison.png
- **Location in content.md**: After line 2012 (after plt.savefig call)
- **Description**: Two-panel comparison of quantization methods:
  - Left: Accuracy vs Speedup scatter plot showing trade-offs between different quantization techniques
  - Right: Memory footprint bar chart comparing memory usage across methods
- **Suggested markdown insertion**:
  ```markdown
  ![Quantization Methods Comparison: Accuracy vs Speedup and Memory Usage](diagrams/quantization_comparison.png)
  ```

## Image Specifications

All diagrams follow the textbook style guidelines:
- **Resolution**: 150 DPI
- **Max width**: ~800px (14 inches at 150 DPI for wide figures, 10 inches for square)
- **Color palette**:
  - Blue: #2196F3
  - Green: #4CAF50
  - Orange: #FF9800
  - Red: #F44336
  - Purple: #9C27B0
  - Gray: #607D8B
- **Background**: White
- **Font sizes**: Minimum 12pt for readability
- **Layout**: All use tight_layout() for proper spacing

## Regeneration

To regenerate all diagrams, run:
```bash
cd /home/chirag/ds-book/book/course-22/ch64
python generate_diagrams.py
```

## Integration Notes

The content.md file currently contains the Python code to generate these diagrams but does not include the actual image references. To complete the integration:

1. Add markdown image references at the three locations specified above
2. The images are already generated and saved in the diagrams/ directory
3. No modifications to the existing Python code blocks are needed - they serve as documentation of how the diagrams were created
