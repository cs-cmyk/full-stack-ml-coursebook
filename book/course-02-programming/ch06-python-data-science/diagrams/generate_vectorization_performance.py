#!/usr/bin/env python3
"""
Generate vectorization performance comparison diagram
"""
import numpy as np
import matplotlib.pyplot as plt
import time

# Create test data
arr = np.random.randn(1_000_000)

# Benchmark function
def benchmark_operation(name, loop_func, vectorized_func):
    # Loop approach
    start = time.time()
    loop_func()
    loop_time = time.time() - start

    # Vectorized approach
    start = time.time()
    vectorized_func()
    vectorized_time = time.time() - start

    return loop_time, vectorized_time, loop_time / vectorized_time

# Operations to test
operations = {}

# Square
operations['Square'] = benchmark_operation(
    'Square',
    lambda: [x**2 for x in arr],
    lambda: arr ** 2
)

# Square root
operations['Sqrt'] = benchmark_operation(
    'Sqrt',
    lambda: [x**0.5 if x > 0 else 0 for x in arr],
    lambda: np.sqrt(np.abs(arr))
)

# Add constant
operations['Add 10'] = benchmark_operation(
    'Add 10',
    lambda: [x + 10 for x in arr],
    lambda: arr + 10
)

# Multiply
operations['Multiply by 2'] = benchmark_operation(
    'Multiply by 2',
    lambda: [x * 2 for x in arr],
    lambda: arr * 2
)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
names = list(operations.keys())
loop_times = [operations[name][0] * 1000 for name in names]  # Convert to ms
vectorized_times = [operations[name][1] * 1000 for name in names]  # Convert to ms
speedups = [operations[name][2] for name in names]

x = np.arange(len(names))
width = 0.35

# Use consistent color palette
bars1 = ax.bar(x - width/2, loop_times, width, label='Python Loop', color='#F44336')
bars2 = ax.bar(x + width/2, vectorized_times, width, label='NumPy Vectorized', color='#4CAF50')

# Add speedup annotations
for i, (bar, speedup) in enumerate(zip(bars2, speedups)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{speedup:.0f}×\nfaster',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Time (milliseconds, log scale)', fontsize=12)
ax.set_xlabel('Operation', fontsize=12)
ax.set_title('Vectorization Performance: Python Loops vs NumPy\n(1 Million Elements)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend(fontsize=11)
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-02-programming/ch06-python-data-science/diagrams/vectorization_performance.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# Output summary
print("Performance Comparison (1 million elements):")
print("-" * 60)
for name in names:
    loop_t, vec_t, speedup = operations[name]
    print(f"{name:15} | Loop: {loop_t*1000:6.1f}ms | NumPy: {vec_t*1000:5.2f}ms | {speedup:4.0f}× faster")

print("\n✓ Diagram saved: vectorization_performance.png")
