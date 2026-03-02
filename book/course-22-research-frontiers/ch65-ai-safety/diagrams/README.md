# Chapter 65 Diagrams

This directory contains all educational diagrams for Chapter 65: The Alignment Problem.

## Generated Diagrams

1. **alignment_problems.png** (156 KB, 2082×878px)
   - Two-panel visualization showing outer vs inner alignment
   - Left: Outer alignment - gap between specified reward R and true values V
   - Right: Inner alignment - learned objective R' diverges from true objective R outside training distribution

2. **reward_hacking_demo.png** (74 KB, 2062×899px)
   - Gridworld demonstration of reward hacking
   - Left: Proper reward specification - agent reaches intended goal
   - Right: Poorly specified reward - agent exploits shortcut state

3. **goal_misgeneralization.png** (145 KB, 2084×1488px)
   - Four-panel visualization of spurious correlations
   - Shows how models learn wrong objectives during training that fail in deployment
   - Compares training vs deployment distributions and model performance

4. **preference_learning.png** (132 KB, 2085×879px)
   - Bradley-Terry model for learning from human preferences (RLHF foundation)
   - Left: Learned vs true reward weights
   - Right: Preference predictions on test set

5. **gridworld_comparison.png** (246 KB, 2085×1477px)
   - Four-panel comparison of three reward function designs
   - Shows learning curves, success rates, state values, and final performance
   - Demonstrates alignment-capability tradeoffs

## Color Palette

All diagrams use a consistent color scheme:
- Blue (#2196F3): Primary/training data
- Green (#4CAF50, #2E7D32): Correct/true values
- Orange (#FF9800): Warnings/alternatives
- Red (#F44336, #C62828): Errors/misalignment
- Purple (#9C27B0): Special features
- Gray (#607D8B): Neutral elements

## Generation Scripts

Python scripts to regenerate each diagram are included:
- `generate_alignment_problems.py`
- `generate_reward_hacking.py`
- `generate_goal_misgeneralization.py`
- `generate_preference_learning.py`
- `generate_gridworld_comparison.py`

All diagrams are saved at 150 DPI with tight bounding boxes for optimal textbook presentation.
