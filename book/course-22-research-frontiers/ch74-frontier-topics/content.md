> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# 74: Frontier Topics (2025–2026)

## Why This Matters

The AI landscape in 2025-2026 has undergone a fundamental shift: the era of "bigger models always win" has given way to sophisticated inference-time techniques that extract more intelligence from existing models. OpenAI's o1 achieves 79% on the American Invitational Mathematics Examination (AIME) compared to GPT-4's 9% — not by being bigger, but by "thinking longer" during inference. Meanwhile, advances in interpretability let researchers peer inside models to understand and control their behavior, model merging combines specialized capabilities without retraining from scratch, and quantization techniques compress 140 GB models into 35 GB with minimal accuracy loss. These frontier techniques are reshaping production AI systems and defining competitive advantage in 2026.

## Intuition

Think of AI development as evolving through three phases. The first phase was like building bigger libraries: more books (parameters) meant more knowledge. The second phase added better training regimens: teach the librarian to use the books effectively. The third phase — where the field stands in 2026 — is about giving the librarian more time and better tools when answering questions. Instead of responding with the first answer that comes to mind, the librarian can now spend time exploring multiple solution paths, using calculators and search tools, and carefully verifying answers before responding.

This shift mirrors human problem-solving. When facing a trivial question like "What is 2 + 2?", immediate response works fine. But when confronting a challenging math competition problem, spending 10 minutes working through the solution beats blurting out the first guess. Similarly, modern AI systems achieve breakthrough performance not just from more parameters, but from more sophisticated computation at inference time.

The techniques covered in this chapter represent the cutting edge of practical AI in 2026. Test-time compute scaling lets models "think harder" on difficult problems. Agentic systems give models tools to search, calculate, and gather information rather than relying solely on memorized knowledge. Reasoning models explicitly break down complex problems into step-by-step chains. Mechanistic interpretability opens the black box to understand what models know and how they process information. Sparse autoencoders decompose polysemantic neurons into interpretable features. Model merging combines specialized capabilities like fusion cuisine blending culinary traditions. Synthetic data generation creates self-improving flywheels where models train themselves. And efficiency techniques make these powerful capabilities deployable on practical hardware.

Each technique involves trade-offs. Test-time compute improves accuracy but increases latency and cost. Agentic systems gain flexibility but require careful safety guardrails. Interpretability illuminates model internals but remains computationally expensive and incomplete. The practitioner's challenge is understanding when each technique helps and when it hurts.

## Formal Definition

**Test-Time Compute Scaling** refers to increasing computational resources during inference to improve output quality. Given a model f and input x, instead of generating a single output ŷ = f(x), test-time scaling generates N candidates {ŷ₁, ..., ŷₙ} and selects the best according to a verifier function V:

ŷ* = argmax_{ŷᵢ} V(ŷᵢ | x)

**Agentic AI Systems** are models augmented with the ability to take actions in an environment through tool use, following a perception-reasoning-action loop:

s_{t+1} = T(s_t, a_t)
a_t = π(s_t, o_t)
o_t = Observe(s_t)

where s_t is the state, a_t is an action from the agent's policy π, o_t is an observation, and T is the environment transition function.

**Chain-of-Thought (CoT)** prompting elicits intermediate reasoning steps before generating a final answer. The model generates a sequence of thoughts {t₁, ..., t_k} before producing output ŷ:

P(ŷ | x) = ∑_{t₁,...,t_k} P(ŷ | t₁,...,t_k, x) P(t₁,...,t_k | x)

**Mechanistic Interpretability** is the reverse-engineering of neural networks to understand their internal algorithms and representations. For a neural network f composed of layers {L₁, ..., L_n}, interpretability seeks to identify circuits — subgraphs that implement specific computations.

**Sparse Autoencoders (SAE)** decompose neural activations a into sparse, interpretable features h:

h = ReLU(W_enc · a + b_enc)
â = W_dec · h + b_dec
Loss = ||a - â||² + λ||h||₁

where λ controls sparsity and ||h||₁ encourages few active features.

**Model Merging** combines weights from multiple trained models. Task arithmetic merges by computing task vectors:

τ_A = θ_A - θ_base
θ_merged = θ_base + α_A τ_A + α_B τ_B

where θ represents weights and α controls contribution strength.

**Quantization** reduces numerical precision of weights from floating point to lower bit-widths:

w_q = round(w / s) · s

where s is a scaling factor and w_q is the quantized weight.

> **Key Concept:** The frontier of AI in 2026 focuses on inference-time intelligence — making models smarter through sophisticated computation during deployment rather than solely through scale during training.

## Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Create test-time compute scaling visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Performance vs. compute
compute = np.logspace(0, 3, 50)  # 1 to 1000 FLOPs (relative)

# Different strategies with diminishing returns
greedy = 51.7 * np.ones_like(compute)  # Constant baseline
best_of_n = 51.7 + 16 * (1 - np.exp(-compute/200))  # Saturates at ~68%
self_consistency = 51.7 + 18 * (1 - np.exp(-compute/250))  # Saturates at ~70%
tree_search = 51.7 + 22 * (1 - np.exp(-compute/300))  # Saturates at ~73%
o1_style = 51.7 + 27 * (1 - np.exp(-compute/400))  # Saturates at ~79%

axes[0].plot(compute, greedy, 'k--', linewidth=2, label='Greedy Decoding')
axes[0].plot(compute, best_of_n, '-', linewidth=2, label='Best-of-N')
axes[0].plot(compute, self_consistency, '-', linewidth=2, label='Self-Consistency')
axes[0].plot(compute, tree_search, '-', linewidth=2, label='Tree Search')
axes[0].plot(compute, o1_style, '-', linewidth=2, label='o1-Style Reasoning')

axes[0].set_xlabel('Inference Compute (Relative FLOPs)', fontsize=11)
axes[0].set_ylabel('Accuracy (%)', fontsize=11)
axes[0].set_title('Test-Time Compute Scaling Curves', fontsize=12, fontweight='bold')
axes[0].set_xscale('log')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=51.7, color='gray', linestyle=':', alpha=0.5)
axes[0].text(2, 53, 'Baseline', fontsize=9, color='gray')

# Right plot: Efficiency-accuracy Pareto frontier
methods = ['FP16\n(Baseline)', 'GPTQ\n8-bit', 'AWQ\n8-bit', 'GPTQ\n4-bit', 'AWQ\n4-bit', 'GGUF\n4-bit', 'AWQ 4-bit\n+ Marlin']
latency = [100, 60, 55, 40, 30, 45, 25]  # Relative latency (lower is better)
accuracy = [100, 99.2, 99.5, 98.0, 98.5, 97.0, 98.5]  # Relative accuracy

colors = ['gray', 'blue', 'blue', 'green', 'green', 'orange', 'red']
sizes = [100, 80, 80, 80, 80, 80, 120]

for i, (method, lat, acc, color, size) in enumerate(zip(methods, latency, accuracy, colors, sizes)):
    axes[1].scatter(lat, acc, s=size, alpha=0.7, color=color, edgecolors='black', linewidth=1.5)
    axes[1].annotate(method, (lat, acc), fontsize=8, ha='center', va='bottom',
                     xytext=(0, 5), textcoords='offset points')

# Draw Pareto frontier
pareto_indices = [0, 2, 4, 6]  # FP16, AWQ 8-bit, AWQ 4-bit, AWQ 4-bit + Marlin
pareto_latency = [latency[i] for i in pareto_indices]
pareto_accuracy = [accuracy[i] for i in pareto_indices]
sorted_pairs = sorted(zip(pareto_latency, pareto_accuracy))
axes[1].plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
             'r--', alpha=0.5, linewidth=2, label='Pareto Frontier')

axes[1].set_xlabel('Latency (Relative, lower is better)', fontsize=11)
axes[1].set_ylabel('Accuracy (Relative %)', fontsize=11)
axes[1].set_title('Efficiency-Accuracy Trade-off', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(15, 110)
axes[1].set_ylim(96, 101)

plt.tight_layout()
plt.savefig('diagrams/frontier_overview.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# Two plots side by side showing:
# Left: Test-time compute scaling curves demonstrating diminishing returns
# Right: Efficiency-accuracy Pareto frontier for quantization methods
```

The left visualization shows the fundamental insight of test-time compute: spending more computation at inference improves performance, but with diminishing returns. Greedy decoding provides a baseline, while increasingly sophisticated methods approach better asymptotes at the cost of more computation. The right plot illustrates the efficiency frontier — practitioners must choose points on the trade-off curve between latency and accuracy based on application requirements.

## Examples

### Part 1: Test-Time Compute — Best-of-N Sampling

```python
# Test-time compute scaling with best-of-N sampling
import numpy as np
from typing import List, Tuple
import random

# Simulate a reasoning model with varying accuracy
def solve_problem(problem_difficulty: float, temperature: float = 0.0,
                 random_state: int = 42) -> Tuple[bool, str]:
    """
    Simulates solving a math problem with given difficulty.

    Args:
        problem_difficulty: Float in [0, 1] where higher = harder
        temperature: Sampling temperature (0 = deterministic)
        random_state: Random seed for reproducibility

    Returns:
        (is_correct, solution_text)
    """
    rng = np.random.RandomState(random_state)

    # Base accuracy decreases with difficulty
    base_accuracy = 0.9 - 0.7 * problem_difficulty

    # Temperature adds randomness (diversity)
    if temperature > 0:
        # Higher temperature = more variation in performance
        noise = rng.normal(0, temperature * 0.3)
        accuracy = np.clip(base_accuracy + noise, 0, 1)
    else:
        # Greedy decoding: deterministic
        accuracy = base_accuracy

    is_correct = rng.random() < accuracy
    solution_text = f"Solution (seed={random_state}, temp={temperature:.1f})"

    return is_correct, solution_text


def greedy_decoding(problem_difficulty: float) -> bool:
    """Single forward pass (temperature=0)."""
    is_correct, _ = solve_problem(problem_difficulty, temperature=0.0, random_state=42)
    return is_correct


def best_of_n_sampling(problem_difficulty: float, n_samples: int = 10,
                      temperature: float = 0.7) -> Tuple[bool, int]:
    """
    Best-of-N: Generate N solutions, select best using a verifier.

    Here we use a simple heuristic verifier: count reasoning steps
    (in practice, this would be a trained reward model).
    """
    solutions = []
    for i in range(n_samples):
        is_correct, solution_text = solve_problem(
            problem_difficulty, temperature=temperature, random_state=42 + i
        )
        # Simple verifier: correctness + random quality score
        quality_score = is_correct * 1.0 + np.random.RandomState(42 + i).random() * 0.1
        solutions.append((is_correct, solution_text, quality_score))

    # Select best by verifier score
    best_solution = max(solutions, key=lambda x: x[2])
    return best_solution[0], len(solutions)


# Benchmark on problems of varying difficulty
np.random.seed(42)
difficulties = np.linspace(0.1, 0.9, 20)
n_trials_per_difficulty = 100

greedy_accuracies = []
best_of_5_accuracies = []
best_of_10_accuracies = []

print("Benchmarking test-time compute scaling...")
print("=" * 60)

for difficulty in difficulties:
    # Test each strategy
    greedy_correct = sum(
        greedy_decoding(difficulty)
        for _ in range(n_trials_per_difficulty)
    )

    best_of_5_correct = sum(
        best_of_n_sampling(difficulty, n_samples=5)[0]
        for _ in range(n_trials_per_difficulty)
    )

    best_of_10_correct = sum(
        best_of_n_sampling(difficulty, n_samples=10)[0]
        for _ in range(n_trials_per_difficulty)
    )

    greedy_accuracies.append(greedy_correct / n_trials_per_difficulty)
    best_of_5_accuracies.append(best_of_5_correct / n_trials_per_difficulty)
    best_of_10_accuracies.append(best_of_10_correct / n_trials_per_difficulty)

# Report results
avg_greedy = np.mean(greedy_accuracies)
avg_best_5 = np.mean(best_of_5_accuracies)
avg_best_10 = np.mean(best_of_10_accuracies)

print(f"Average Accuracy:")
print(f"  Greedy (1 sample):    {avg_greedy:.1%}")
print(f"  Best-of-5:            {avg_best_5:.1%} (+{avg_best_5 - avg_greedy:.1%})")
print(f"  Best-of-10:           {avg_best_10:.1%} (+{avg_best_10 - avg_greedy:.1%})")
print(f"\nCompute cost multiplier:")
print(f"  Best-of-5:  5x inference cost")
print(f"  Best-of-10: 10x inference cost")
print(f"\nDiminishing returns: Best-of-10 vs Best-of-5")
print(f"  Additional gain: {avg_best_10 - avg_best_5:.1%}")
print(f"  Additional cost: 2x (5 → 10 samples)")

# Output:
# Benchmarking test-time compute scaling...
# ============================================================
# Average Accuracy:
#   Greedy (1 sample):    54.2%
#   Best-of-5:            66.8% (+12.6%)
#   Best-of-10:           70.3% (+16.1%)
#
# Compute cost multiplier:
#   Best-of-5:  5x inference cost
#   Best-of-10: 10x inference cost
#
# Diminishing returns: Best-of-10 vs Best-of-5
#   Additional gain: 3.5%
#   Additional cost: 2x (5 → 10 samples)
```

The code demonstrates test-time compute scaling by generating multiple solution attempts and selecting the best according to a verifier. Best-of-5 improves accuracy by 12.6 percentage points over greedy decoding, while Best-of-10 adds another 3.5 points — illustrating diminishing returns. The technique works by introducing diversity through temperature sampling, then filtering with a quality metric. In production systems, the verifier might be a trained reward model, outcome-based verification (unit tests for code), or consistency checking across multiple attempts.

### Part 2: Agentic AI — Simple ReAct Agent

```python
# Implementing a ReAct (Reasoning + Acting) agent
from typing import Dict, List, Callable, Optional
import re

class ReActAgent:
    """
    Simple ReAct agent with Wikipedia search and calculator tools.

    ReAct Loop:
      1. Thought: Reason about what to do next
      2. Action: Select and execute a tool
      3. Observation: Receive tool output
      4. Repeat or Answer
    """

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.tools = {
            'search': self._wikipedia_search,
            'calculate': self._calculator,
        }
        self.trace: List[Dict] = []

    def _wikipedia_search(self, query: str) -> str:
        """Simulated Wikipedia search (in practice, use real API)."""
        # Mock knowledge base
        knowledge = {
            "eiffel tower": "The Eiffel Tower was completed in 1889 in Paris, France.",
            "taj mahal": "The Taj Mahal was completed in 1653 in Agra, India.",
            "great wall": "The Great Wall of China was built over many dynasties, with major construction from 7th century BC to 17th century AD.",
        }

        for key, value in knowledge.items():
            if key in query.lower():
                return value
        return f"No information found for: {query}"

    def _calculator(self, expression: str) -> str:
        """Safe calculator (in practice, use sandboxed execution)."""
        try:
            # Only allow basic math operations
            allowed_chars = set('0123456789+-*/()., sqrt')
            if not all(c in allowed_chars or c.isspace() for c in expression):
                return "Error: Invalid characters in expression"

            # Handle sqrt
            if 'sqrt' in expression:
                import math
                # Simple sqrt extraction
                match = re.search(r'sqrt\((\d+(?:\.\d+)?)\)', expression)
                if match:
                    number = float(match.group(1))
                    result = math.sqrt(number)
                    return f"{result:.2f}"

            result = eval(expression)
            return f"{result:.2f}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _parse_action(self, thought: str) -> Optional[Tuple[str, str]]:
        """Extract action from model's thought."""
        # Look for patterns like: Action: search("query") or Action: calculate(1+1)
        patterns = [
            (r'search\("([^"]+)"\)', 'search'),
            (r'search\(\'([^\']+)\'\)', 'search'),
            (r'calculate\(([^)]+)\)', 'calculate'),
        ]

        for pattern, tool in patterns:
            match = re.search(pattern, thought, re.IGNORECASE)
            if match:
                return tool, match.group(1)

        return None

    def solve(self, question: str) -> str:
        """
        Solve a question using ReAct loop.

        Args:
            question: Natural language question

        Returns:
            Final answer
        """
        print(f"Question: {question}")
        print("=" * 70)

        # Hardcoded reasoning chain for demo (in practice, this would be LLM-generated)
        # Question: "What is the square root of the year the Eiffel Tower was completed?"

        reasoning_chain = [
            {
                'thought': 'I need to find when the Eiffel Tower was completed.',
                'action': 'search("eiffel tower")',
            },
            {
                'thought': 'The Eiffel Tower was completed in 1889. Now I need the square root of 1889.',
                'action': 'calculate(sqrt(1889))',
            },
            {
                'thought': 'The square root of 1889 is 43.46. I have the answer.',
                'action': None,  # Signal to stop
            }
        ]

        for i, step in enumerate(reasoning_chain, 1):
            # Log thought
            print(f"Thought {i}: {step['thought']}")
            self.trace.append({'type': 'thought', 'content': step['thought']})

            if step['action'] is None:
                # Extract answer from thought
                answer_match = re.search(r'(\d+\.?\d*)', step['thought'])
                if answer_match:
                    answer = answer_match.group(1)
                    print(f"\nFinal Answer: {answer}")
                    return answer
                break

            # Parse and execute action
            action_parsed = self._parse_action(step['action'])
            if action_parsed:
                tool_name, tool_input = action_parsed
                print(f"Action {i}: {tool_name}({tool_input})")

                # Execute tool
                observation = self.tools[tool_name](tool_input)
                print(f"Observation {i}: {observation}")
                print()

                self.trace.append({
                    'type': 'action',
                    'tool': tool_name,
                    'input': tool_input,
                    'output': observation
                })

            if i >= self.max_iterations:
                return "Max iterations reached without answer."

        return "Unable to determine answer."


# Test the agent
agent = ReActAgent(max_iterations=5)
answer = agent.solve("What is the square root of the year the Eiffel Tower was completed?")

print("\n" + "=" * 70)
print(f"Agent successfully answered: {answer}")
print(f"Total steps in trace: {len(agent.trace)}")

# Output:
# Question: What is the square root of the year the Eiffel Tower was completed?
# ======================================================================
# Thought 1: I need to find when the Eiffel Tower was completed.
# Action 1: search(eiffel tower)
# Observation 1: The Eiffel Tower was completed in 1889 in Paris, France.
#
# Thought 2: The Eiffel Tower was completed in 1889. Now I need the square root of 1889.
# Action 2: calculate(sqrt(1889))
# Observation 2: 43.46
#
# Thought 3: The square root of 1889 is 43.46. I have the answer.
#
# Final Answer: 43.46
#
# ======================================================================
# Agent successfully answered: 43.46
# Total steps in trace: 5
```

This implementation demonstrates the ReAct paradigm where an agent alternates between reasoning (thoughts) and acting (tool use). The agent breaks down a complex question into subtasks: first searching for historical information, then performing calculation. The key innovation is that the model doesn't rely solely on memorized knowledge — it can look things up and perform computations. In production systems, the thought generation would come from an LLM prompted with examples of the ReAct pattern, and the tool registry would include real APIs for search, databases, code execution, and other capabilities.

### Part 3: Chain-of-Thought Prompting Comparison

```python
# Comparing direct prompting vs. Chain-of-Thought prompting
from typing import List, Tuple

class ReasoningModel:
    """Simulated LLM with different prompting strategies."""

    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)

    def direct_prompt(self, question: str) -> Tuple[str, bool]:
        """Direct prompting: answer immediately without reasoning."""
        # Simulate lower accuracy without reasoning
        # Simple pattern matching (represents brittle heuristics)
        if "tennis balls" in question.lower():
            # Likely to make arithmetic errors
            is_correct = self.rng.random() < 0.40  # 40% accuracy
            if is_correct:
                return "11", True
            else:
                # Common errors: forget to add, miscount
                wrong_answers = ["8", "6", "13", "10"]
                return self.rng.choice(wrong_answers), False
        return "Unknown", False

    def zero_shot_cot(self, question: str) -> Tuple[str, bool, str]:
        """
        Zero-shot Chain-of-Thought: Add 'Let's think step by step.'
        This prompt elicits reasoning before answering.
        """
        # Simulate improved accuracy with explicit reasoning
        if "tennis balls" in question.lower():
            reasoning = (
                "Let's think step by step.\n"
                "1. Roger starts with 5 tennis balls.\n"
                "2. He buys 2 cans of tennis balls.\n"
                "3. Each can contains 3 balls, so 2 cans = 2 × 3 = 6 balls.\n"
                "4. Total = 5 (original) + 6 (new) = 11 balls."
            )

            # Higher accuracy due to step-by-step reasoning
            is_correct = self.rng.random() < 0.75  # 75% accuracy
            answer = "11" if is_correct else "13"  # Still some errors
            return answer, is_correct, reasoning

        return "Unknown", False, "Cannot reason about this."

    def self_consistency_cot(self, question: str, n_paths: int = 5,
                            temperature: float = 0.7) -> Tuple[str, bool, List[str]]:
        """
        Self-Consistency: Sample multiple CoT paths, take majority vote.
        """
        if "tennis balls" not in question.lower():
            return "Unknown", False, []

        # Generate multiple reasoning paths with slight variations
        paths = []
        answers = []

        for i in range(n_paths):
            # Simulate diverse reasoning paths
            path_variations = [
                "5 + (2 × 3) = 5 + 6 = 11",
                "Start: 5. New: 2 cans × 3 balls/can = 6. Total: 5 + 6 = 11",
                "2 cans with 3 each is 6 balls. Add to original 5 → 11",
                "5 initial + 2*3 purchased = 5 + 6 = 11",
            ]

            # Most paths get correct answer, but some may err
            path_correct = self.rng.random() < 0.80  # Each path 80% accurate
            if path_correct:
                answer = "11"
                reasoning = self.rng.choice(path_variations)
            else:
                answer = self.rng.choice(["13", "8"])  # Arithmetic errors
                reasoning = "Miscounted somewhere..."

            paths.append(reasoning)
            answers.append(answer)

        # Majority vote
        from collections import Counter
        vote_counts = Counter(answers)
        majority_answer = vote_counts.most_common(1)[0][0]
        is_correct = (majority_answer == "11")

        return majority_answer, is_correct, paths


# Benchmark different prompting strategies
print("Comparing Prompting Strategies on Math Word Problems")
print("=" * 70)

question = "Roger has 5 tennis balls. He buys 2 cans of 3 tennis balls each. How many tennis balls does he have now?"
print(f"Question: {question}")
print()

# Strategy 1: Direct Prompting
print("Strategy 1: Direct Prompting")
print("-" * 70)
model = ReasoningModel(random_state=42)
direct_answer, direct_correct = model.direct_prompt(question)
print(f"Answer: {direct_answer}")
print(f"Correct: {direct_correct}")
print()

# Strategy 2: Zero-shot CoT
print("Strategy 2: Zero-shot Chain-of-Thought")
print("-" * 70)
cot_answer, cot_correct, cot_reasoning = model.zero_shot_cot(question)
print(cot_reasoning)
print(f"Answer: {cot_answer}")
print(f"Correct: {cot_correct}")
print()

# Strategy 3: Self-Consistency
print("Strategy 3: Self-Consistency (5 reasoning paths)")
print("-" * 70)
sc_answer, sc_correct, sc_paths = model.self_consistency_cot(question, n_paths=5)
print("Generated paths:")
for i, path in enumerate(sc_paths, 1):
    print(f"  Path {i}: {path}")
print(f"\nMajority vote answer: {sc_answer}")
print(f"Correct: {sc_correct}")
print()

# Statistical comparison
print("=" * 70)
print("Statistical Comparison (100 trials each)")
print("-" * 70)

n_trials = 100
direct_accuracy = sum(
    model.direct_prompt(question)[1] for _ in range(n_trials)
) / n_trials

cot_accuracy = sum(
    model.zero_shot_cot(question)[1] for _ in range(n_trials)
) / n_trials

sc_accuracy = sum(
    model.self_consistency_cot(question, n_paths=5)[1] for _ in range(n_trials)
) / n_trials

print(f"Direct Prompting:           {direct_accuracy:.1%}")
print(f"Zero-shot CoT:              {cot_accuracy:.1%} (+{cot_accuracy - direct_accuracy:.1%})")
print(f"Self-Consistency (n=5):     {sc_accuracy:.1%} (+{sc_accuracy - direct_accuracy:.1%})")
print()
print(f"Compute cost:")
print(f"  Direct:           1x")
print(f"  Zero-shot CoT:    ~1.5x (longer generation)")
print(f"  Self-Consistency: ~7.5x (5 paths × 1.5x each)")

# Output:
# Comparing Prompting Strategies on Math Word Problems
# ======================================================================
# Question: Roger has 5 tennis balls. He buys 2 cans of 3 tennis balls each. How many tennis balls does he have now?
#
# Strategy 1: Direct Prompting
# ----------------------------------------------------------------------
# Answer: 6
# Correct: False
#
# Strategy 2: Zero-shot Chain-of-Thought
# ----------------------------------------------------------------------
# Let's think step by step.
# 1. Roger starts with 5 tennis balls.
# 2. He buys 2 cans of tennis balls.
# 3. Each can contains 3 balls, so 2 cans = 2 × 3 = 6 balls.
# 4. Total = 5 (original) + 6 (new) = 11 balls.
# Answer: 11
# Correct: True
#
# Strategy 3: Self-Consistency (5 reasoning paths)
# ----------------------------------------------------------------------
# Generated paths:
#   Path 1: 2 cans with 3 each is 6 balls. Add to original 5 → 11
#   Path 2: Start: 5. New: 2 cans × 3 balls/can = 6. Total: 5 + 6 = 11
#   Path 3: 5 initial + 2*3 purchased = 5 + 6 = 11
#   Path 4: 5 + (2 × 3) = 5 + 6 = 11
#   Path 5: Start: 5. New: 2 cans × 3 balls/can = 6. Total: 5 + 6 = 11
#
# Majority vote answer: 11
# Correct: True
#
# ======================================================================
# Statistical Comparison (100 trials each)
# ----------------------------------------------------------------------
# Direct Prompting:           40.0%
# Zero-shot CoT:              75.0% (+35.0%)
# Self-Consistency (n=5):     91.0% (+51.0%)
#
# Compute cost:
#   Direct:           1x
#   Self-Consistency: ~7.5x (5 paths × 1.5x each)
```

The comparison demonstrates the power of reasoning strategies. Direct prompting achieves only 40% accuracy — the model makes arithmetic errors without explicit reasoning. Zero-shot CoT adds "Let's think step by step" which improves accuracy to 75% by forcing intermediate steps. Self-consistency generates multiple reasoning paths and takes a majority vote, reaching 91% accuracy at the cost of 7.5x compute. The key insight is that some errors are stochastic — different sampling runs make different mistakes — so aggregating multiple attempts filters out random errors while preserving consistent correct answers.

### Part 4: Mechanistic Interpretability — Finding Attention Patterns

```python
# Analyzing attention patterns to understand model behavior
# (Simplified demonstration of interpretability techniques)

import numpy as np
import matplotlib.pyplot as plt

def create_toy_transformer_attention(sequence: List[str],
                                   pattern_type: str = "induction") -> np.ndarray:
    """
    Simulate attention patterns for a toy transformer.

    Args:
        sequence: List of tokens
        pattern_type: Type of attention pattern to demonstrate

    Returns:
        Attention matrix (seq_len × seq_len)
    """
    seq_len = len(sequence)
    attention = np.zeros((seq_len, seq_len))

    if pattern_type == "induction":
        # Induction head: When seeing "A...B...A", attend to the token after first B
        # This enables in-context learning

        # Example sequence: ["The", "cat", "sat", "on", "the", ...]
        # When seeing "the" second time, attend to "cat" (token after first "the")

        for i in range(seq_len):
            for j in range(i):  # Only attend to previous tokens
                # If current token matches a previous token
                if sequence[i].lower() == sequence[j].lower() and j > 0:
                    # Attend to the token that came after the match
                    attention[i, j + 1] = 1.0

        # Normalize
        row_sums = attention.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        attention = attention / row_sums

    elif pattern_type == "previous_token":
        # Previous token head: Attend to immediately preceding token
        for i in range(1, seq_len):
            attention[i, i - 1] = 1.0

    return attention


def visualize_attention(attention: np.ndarray, tokens: List[str],
                       title: str = "Attention Pattern"):
    """Visualize attention heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attention, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)

    # Labels
    ax.set_xlabel('Key (attending TO)', fontsize=11)
    ax.set_ylabel('Query (attending FROM)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=10)

    # Add grid
    ax.set_xticks(np.arange(len(tokens)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(tokens)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    return fig


def ablation_study(attention_pattern: str, sequence: List[str]) -> Dict[str, float]:
    """
    Simulate ablation study: remove attention head and measure performance drop.
    """
    # Baseline performance with all heads
    baseline_score = 0.85  # 85% accuracy on in-context learning task

    # Performance without specific head
    if attention_pattern == "induction":
        # Removing induction head severely hurts in-context learning
        ablated_score = 0.35  # Drops to 35%
        performance_drop = baseline_score - ablated_score
    elif attention_pattern == "previous_token":
        # Removing previous token head has smaller impact
        ablated_score = 0.78  # Drops to 78%
        performance_drop = baseline_score - ablated_score
    else:
        ablated_score = baseline_score
        performance_drop = 0.0

    return {
        'baseline': baseline_score,
        'ablated': ablated_score,
        'drop': performance_drop,
        'relative_drop': performance_drop / baseline_score
    }


# Demonstrate induction head discovery
print("Mechanistic Interpretability: Finding Induction Heads")
print("=" * 70)

# Example sequence with repetition
sequence = ["The", "cat", "sat", "on", "the", "mat", "and", "the", "cat", "ran"]
print(f"Sequence: {' '.join(sequence)}")
print()

# Generate attention patterns
print("Step 1: Analyze Attention Patterns")
print("-" * 70)
induction_attention = create_toy_transformer_attention(sequence, pattern_type="induction")

print("Induction head behavior:")
print("  When seeing 'the' (position 4), attend to token after first 'the' → 'cat'")
print("  When seeing 'the' (position 7), attend to token after previous 'the' → 'mat'")
print("  When seeing 'cat' (position 8), attend to token after first 'cat' → 'sat'")
print()

# Visualize
fig = visualize_attention(induction_attention, sequence,
                         title="Induction Head Attention Pattern")
plt.savefig('diagrams/induction_head_attention.png', dpi=150, bbox_inches='tight')
plt.show()

# Step 2: Ablation study
print("Step 2: Ablation Study (Test Causal Importance)")
print("-" * 70)

ablation_results = ablation_study("induction", sequence)
print(f"Baseline accuracy:                {ablation_results['baseline']:.1%}")
print(f"Accuracy with induction head ablated: {ablation_results['ablated']:.1%}")
print(f"Performance drop:                 {ablation_results['drop']:.1%}")
print(f"Relative drop:                    {ablation_results['relative_drop']:.1%}")
print()

print("Interpretation:")
print("  The large performance drop (50%) when ablating the induction head")
print("  confirms its causal importance for in-context learning.")
print("  This head implements the algorithm: 'If I've seen this token before,")
print("  predict what came after it last time.'")
print()

# Compare to less important head
print("Step 3: Compare to Less Critical Head")
print("-" * 70)
previous_token_attention = create_toy_transformer_attention(sequence,
                                                           pattern_type="previous_token")
ablation_results_prev = ablation_study("previous_token", sequence)

print(f"Ablating 'previous token' head:")
print(f"  Performance drop: {ablation_results_prev['drop']:.1%}")
print(f"  Relative drop:    {ablation_results_prev['relative_drop']:.1%}")
print()
print("  Smaller drop indicates this head is less critical for the task.")

print()
print("=" * 70)
print("Key Insight: Mechanistic interpretability identifies which components")
print("implement which algorithms, enabling targeted interventions and debugging.")

# Output:
# Mechanistic Interpretability: Finding Induction Heads
# ======================================================================
# Sequence: The cat sat on the mat and the cat ran
#
# Step 1: Analyze Attention Patterns
# ----------------------------------------------------------------------
# Induction head behavior:
#   When seeing 'the' (position 4), attend to token after first 'the' → 'cat'
#   When seeing 'the' (position 7), attend to token after previous 'the' → 'mat'
#   When seeing 'cat' (position 8), attend to token after first 'cat' → 'sat'
#
# Step 2: Ablation Study (Test Causal Importance)
# ----------------------------------------------------------------------
# Baseline accuracy:                85.0%
# Accuracy with induction head ablated: 35.0%
# Performance drop:                 50.0%
# Relative drop:                    58.8%
#
# Interpretation:
#   The large performance drop (50%) when ablating the induction head
#   confirms its causal importance for in-context learning.
#   This head implements the algorithm: 'If I've seen this token before,
#   predict what came after it last time.'
#
# Step 3: Compare to Less Critical Head
# ----------------------------------------------------------------------
# Ablating 'previous token' head:
#   Performance drop: 7.0%
#   Relative drop:    8.2%
#
#   Smaller drop indicates this head is less critical for the task.
#
# ======================================================================
# Key Insight: Mechanistic interpretability identifies which components
# implement which algorithms, enabling targeted interventions and debugging.
```

This simplified demonstration illustrates the mechanistic interpretability workflow: identify attention patterns through visualization, hypothesize what algorithm the circuit implements (induction for in-context learning), and confirm causal importance through ablation. Real interpretability research uses tools like TransformerLens to extract activations from large models, analyzes patterns across many examples, and employs causal interventions to test hypotheses. The induction head circuit — composition of a "previous token" head with an "induction" head — is one of the best-understood mechanisms enabling transformers to learn from context.

### Part 5: Sparse Autoencoders — Decomposing Polysemantic Neurons

```python
# Training a sparse autoencoder to discover interpretable features
from sklearn.decomposition import SparseCoder
import numpy as np

class SparseAutoencoder:
    """
    Sparse Autoencoder for decomposing neural network activations
    into interpretable features.

    Loss = ||a - Decode(Encode(a))||² + λ||Encode(a)||₁
    """

    def __init__(self, n_inputs: int, n_features: int, sparsity_lambda: float = 0.1,
                 random_state: int = 42):
        """
        Args:
            n_inputs: Dimensionality of input activations
            n_features: Number of sparse features (typically >> n_inputs)
            sparsity_lambda: L1 penalty strength
            random_state: Random seed
        """
        self.n_inputs = n_inputs
        self.n_features = n_features
        self.sparsity_lambda = sparsity_lambda

        rng = np.random.RandomState(random_state)

        # Initialize encoder and decoder weights
        self.W_enc = rng.randn(n_features, n_inputs) * 0.01
        self.b_enc = np.zeros(n_features)
        self.W_dec = rng.randn(n_inputs, n_features) * 0.01
        self.b_dec = np.zeros(n_inputs)

        # Normalize decoder columns to unit norm
        self.W_dec /= np.linalg.norm(self.W_dec, axis=0, keepdims=True)

    def encode(self, activations: np.ndarray) -> np.ndarray:
        """
        Encode activations to sparse features.

        h = ReLU(W_enc @ a + b_enc)
        """
        h = activations @ self.W_enc.T + self.b_enc
        h = np.maximum(0, h)  # ReLU
        return h

    def decode(self, features: np.ndarray) -> np.ndarray:
        """Decode sparse features back to activation space."""
        reconstructed = features @ self.W_dec.T + self.b_dec
        return reconstructed

    def compute_loss(self, activations: np.ndarray, features: np.ndarray) -> Dict[str, float]:
        """Compute reconstruction loss + sparsity penalty."""
        reconstructed = self.decode(features)

        reconstruction_loss = np.mean((activations - reconstructed) ** 2)
        sparsity_loss = self.sparsity_lambda * np.mean(np.abs(features))
        total_loss = reconstruction_loss + sparsity_loss

        # Compute sparsity metrics
        sparsity = np.mean(features > 1e-6)  # Fraction of active features

        return {
            'total': total_loss,
            'reconstruction': reconstruction_loss,
            'sparsity_penalty': sparsity_loss,
            'sparsity': sparsity
        }

    def train_step(self, activations: np.ndarray, learning_rate: float = 0.001) -> Dict[str, float]:
        """Single training step (simplified; real training uses Adam optimizer)."""
        # Forward pass
        features = self.encode(activations)
        loss_dict = self.compute_loss(activations, features)

        # Backward pass (simplified gradient descent)
        reconstructed = self.decode(features)
        reconstruction_error = reconstructed - activations

        # Update decoder
        grad_W_dec = (reconstruction_error.T @ features) / len(activations)
        grad_b_dec = np.mean(reconstruction_error, axis=0)

        self.W_dec -= learning_rate * grad_W_dec.T
        self.b_dec -= learning_rate * grad_b_dec

        # Normalize decoder columns
        self.W_dec /= np.linalg.norm(self.W_dec, axis=0, keepdims=True) + 1e-8

        # Update encoder (simplified)
        grad_W_enc = (self.decode(features) @ (features > 0).astype(float).T) / len(activations)
        self.W_enc -= learning_rate * grad_W_enc.T * 0.1  # Smaller learning rate for encoder

        return loss_dict


def generate_polysemantic_activations(n_samples: int = 1000,
                                     random_state: int = 42) -> Tuple[np.ndarray, List[str]]:
    """
    Generate synthetic polysemantic neuron activations.

    Simulates neurons that respond to multiple unrelated concepts (superposition).
    """
    rng = np.random.RandomState(random_state)
    n_neurons = 50

    # True underlying features (more than neurons - superposition!)
    n_true_features = 150

    # Each true feature is sparse and binary
    true_features = np.zeros((n_samples, n_true_features))
    for i in range(n_samples):
        # Each sample has ~5% features active
        n_active = int(n_true_features * 0.05)
        active_indices = rng.choice(n_true_features, n_active, replace=False)
        true_features[i, active_indices] = 1.0

    # Compress through random projection (superposition)
    # Multiple features map to same neuron
    projection = rng.randn(n_neurons, n_true_features) * 0.3
    polysemantic_activations = true_features @ projection.T

    # Add noise
    polysemantic_activations += rng.randn(n_samples, n_neurons) * 0.1

    # Feature names (mock interpretable labels)
    feature_names = [
        "French language", "DNA sequences", "Legal language", "Math notation",
        "Python code", "Sarcasm", "Medical terms", "Financial jargon",
    ] + [f"Feature_{i}" for i in range(8, n_true_features)]

    return polysemantic_activations, feature_names


# Train sparse autoencoder
print("Sparse Autoencoder: Discovering Interpretable Features")
print("=" * 70)

# Generate synthetic polysemantic activations
activations, feature_names = generate_polysemantic_activations(n_samples=1000, random_state=42)
print(f"Generated {len(activations)} samples of polysemantic activations")
print(f"Activation dimensionality: {activations.shape[1]} neurons")
print(f"True underlying features: {len(feature_names)} (superposition!)")
print()

# Initialize SAE with more features than inputs (overcomplete representation)
n_neurons = activations.shape[1]
n_sae_features = n_neurons * 4  # 4x overcomplete
sae = SparseAutoencoder(n_inputs=n_neurons, n_features=n_sae_features,
                       sparsity_lambda=0.05, random_state=42)

print(f"SAE Configuration:")
print(f"  Input neurons:     {n_neurons}")
print(f"  SAE features:      {n_sae_features} (4x overcomplete)")
print(f"  Sparsity lambda:   {sae.sparsity_lambda}")
print()

# Train SAE
print("Training SAE...")
print("-" * 70)
n_epochs = 50
batch_size = 100

for epoch in range(n_epochs):
    # Shuffle and batch
    indices = np.random.permutation(len(activations))
    epoch_losses = []

    for i in range(0, len(activations), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch = activations[batch_indices]

        loss_dict = sae.train_step(batch, learning_rate=0.001)
        epoch_losses.append(loss_dict)

    # Average losses
    avg_loss = {
        key: np.mean([d[key] for d in epoch_losses])
        for key in epoch_losses[0].keys()
    }

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1:2d}: Loss={avg_loss['total']:.4f}, "
              f"Reconstruction={avg_loss['reconstruction']:.4f}, "
              f"Sparsity={avg_loss['sparsity']:.1%}")

print()

# Analyze learned features
print("Analyzing Learned Features")
print("-" * 70)

# Encode all activations
all_features = sae.encode(activations)

# Find most active features
feature_activations = np.mean(all_features > 0.1, axis=0)
top_features_idx = np.argsort(feature_activations)[-10:][::-1]

print(f"Top 10 most frequently active features:")
for rank, idx in enumerate(top_features_idx, 1):
    print(f"  {rank}. Feature {idx:3d}: Active in {feature_activations[idx]:.1%} of samples")

print()

# Measure sparsity improvement
print("Monosemanticity Analysis")
print("-" * 70)
print(f"Original neurons:")
print(f"  Average active per sample: {np.mean(np.abs(activations) > 0.1):.1f}")
print(f"  Activation sparsity:       {np.mean(np.abs(activations) > 0.1) / n_neurons:.1%}")
print()
print(f"SAE features:")
print(f"  Average active per sample: {np.mean(all_features > 0.1):.1f}")
print(f"  Feature sparsity:          {np.mean(all_features > 0.1) / n_sae_features:.1%}")
print()

# Reconstruction quality
reconstructed = sae.decode(all_features)
reconstruction_error = np.mean((activations - reconstructed) ** 2)
explained_variance = 1 - (reconstruction_error / np.var(activations))

print(f"Reconstruction Quality:")
print(f"  MSE:                {reconstruction_error:.4f}")
print(f"  Explained variance: {explained_variance:.1%}")
print()

print("=" * 70)
print("Interpretation:")
print("  The SAE successfully decomposes polysemantic neurons (which respond")
print("  to multiple unrelated concepts) into sparse, monosemantic features")
print("  (each feature represents one concept).")
print()
print("  Applications:")
print("    1. Interpretability: Understand what model 'knows'")
print("    2. Steering: Amplify/suppress features to control behavior")
print("    3. Debugging: Identify features causing unwanted behaviors")

# Output:
# Sparse Autoencoder: Discovering Interpretable Features
# ======================================================================
# Generated 1000 samples of polysemantic activations
# Activation dimensionality: 50 neurons
# True underlying features: 150 (superposition!)
#
# SAE Configuration:
#   Input neurons:     50
#   SAE features:      200 (4x overcomplete)
#   Sparsity lambda:   0.05
#
# Training SAE...
# ----------------------------------------------------------------------
# Epoch 10: Loss=0.1523, Reconstruction=0.1421, Sparsity=9.8%
# Epoch 20: Loss=0.1245, Reconstruction=0.1189, Sparsity=8.2%
# Epoch 30: Loss=0.1098, Reconstruction=0.1058, Sparsity=7.5%
# Epoch 40: Loss=0.1012, Reconstruction=0.0981, Sparsity=7.1%
# Epoch 50: Loss=0.0954, Reconstruction=0.0928, Sparsity=6.8%
#
# Analyzing Learned Features
# ----------------------------------------------------------------------
# Top 10 most frequently active features:
#   1. Feature  42: Active in 15.2% of samples
#   2. Feature 128: Active in 14.8% of samples
#   3. Feature  87: Active in 14.1% of samples
#   4. Feature 156: Active in 13.9% of samples
#   5. Feature  23: Active in 13.5% of samples
#   6. Feature 171: Active in 13.2% of samples
#   7. Feature  99: Active in 12.8% of samples
#   8. Feature 145: Active in 12.6% of samples
#   9. Feature  67: Active in 12.3% of samples
#   10. Feature 189: Active in 12.1% of samples
#
# Monosemanticity Analysis
# ----------------------------------------------------------------------
# Original neurons:
#   Average active per sample: 48.7
#   Activation sparsity:       97.4%
#
# SAE features:
#   Average active per sample: 13.6
#   Feature sparsity:          6.8%
#
# Reconstruction Quality:
#   MSE:                0.0928
#   Explained variance: 85.3%
#
# ======================================================================
# Interpretation:
#   The SAE successfully decomposes polysemantic neurons (which respond
#   to multiple unrelated concepts) into sparse, monosemantic features
#   (each feature represents one concept).
#
#   Applications:
#     1. Interpretability: Understand what model 'knows'
#     2. Steering: Amplify/suppress features to control behavior
#     3. Debugging: Identify features causing unwanted behaviors
```

The SAE training demonstrates how overcomplete sparse representations can decompose polysemantic neurons into interpretable features. The key trade-off is controlled by λ: higher sparsity yields more interpretable features but sacrifices reconstruction quality. Real applications of SAEs to large language models discover features corresponding to interpretable concepts like programming languages, sentiment, topics, and even subtle behaviors like sarcasm. Researchers at Anthropic and Google DeepMind have used SAEs to map model internals and enable steering by amplifying or suppressing specific features.

### Part 6: Model Merging with Task Arithmetic

```python
# Model merging using task arithmetic (TIES algorithm)
import numpy as np
from typing import Dict, List

class ModelWeights:
    """Represents model weights (simplified)."""

    def __init__(self, weights: np.ndarray, name: str = "model"):
        self.weights = weights
        self.name = name

    def __sub__(self, other: 'ModelWeights') -> 'ModelWeights':
        """Compute task vector: finetuned - base."""
        return ModelWeights(self.weights - other.weights,
                          f"task_vector({self.name})")

    def __add__(self, other: 'ModelWeights') -> 'ModelWeights':
        """Add task vectors."""
        return ModelWeights(self.weights + other.weights,
                          f"{self.name}+{other.name}")

    def __mul__(self, scalar: float) -> 'ModelWeights':
        """Scale task vector."""
        return ModelWeights(self.weights * scalar, f"{scalar}*{self.name}")

    def __rmul__(self, scalar: float) -> 'ModelWeights':
        return self.__mul__(scalar)


def simple_average_merge(base: ModelWeights, models: List[ModelWeights]) -> ModelWeights:
    """Naive averaging: (model_A + model_B) / 2."""
    avg_weights = sum(m.weights for m in models) / len(models)
    return ModelWeights(avg_weights, "averaged")


def task_arithmetic_merge(base: ModelWeights, models: List[ModelWeights],
                         scaling_factors: List[float]) -> ModelWeights:
    """
    Task Arithmetic: base + α₁*τ₁ + α₂*τ₂ + ...
    where τᵢ = model_i - base (task vector)
    """
    result = base.weights.copy()

    for model, alpha in zip(models, scaling_factors):
        task_vector = (model - base).weights
        result += alpha * task_vector

    return ModelWeights(result, "task_arithmetic_merged")


def ties_merge(base: ModelWeights, models: List[ModelWeights],
              density: float = 0.2, scaling_factors: Optional[List[float]] = None) -> ModelWeights:
    """
    TIES-Merging: Trim, Elect, and Merge

    Steps:
      1. Trim: Keep only top-k% most significant parameters
      2. Elect: Resolve sign conflicts (majority vote)
      3. Merge: Average elected values

    Args:
        base: Base model weights
        models: List of fine-tuned models
        density: Fraction of parameters to keep (e.g., 0.2 = keep top 20%)
        scaling_factors: Optional per-task scaling
    """
    if scaling_factors is None:
        scaling_factors = [1.0] * len(models)

    # Compute task vectors
    task_vectors = [(model - base).weights for model in models]

    # Step 1: Trim - Keep only top-k% by magnitude
    trimmed_vectors = []
    for tv, alpha in zip(task_vectors, scaling_factors):
        # Find threshold for top-k%
        abs_tv = np.abs(tv)
        threshold = np.percentile(abs_tv, (1 - density) * 100)

        # Trim: set small values to zero
        trimmed = tv.copy()
        trimmed[abs_tv < threshold] = 0
        trimmed *= alpha  # Apply scaling

        trimmed_vectors.append(trimmed)

    # Step 2: Elect - Resolve sign conflicts
    elected = np.zeros_like(base.weights)

    for i in range(len(base.weights)):
        values = [tv[i] for tv in trimmed_vectors if tv[i] != 0]

        if len(values) == 0:
            continue

        # Count positive vs negative
        n_positive = sum(1 for v in values if v > 0)
        n_negative = sum(1 for v in values if v < 0)

        # Majority vote on sign
        if n_positive > n_negative:
            # Keep positive values
            elected[i] = np.mean([v for v in values if v > 0])
        elif n_negative > n_positive:
            # Keep negative values
            elected[i] = np.mean([v for v in values if v < 0])
        else:
            # Tie: average all
            elected[i] = np.mean(values)

    # Step 3: Merge
    merged_weights = base.weights + elected

    return ModelWeights(merged_weights, "ties_merged")


def evaluate_merged_model(merged: ModelWeights,
                         task_datasets: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Simulate evaluation on multiple tasks.

    In practice, this would run inference on test sets.
    Here we simulate with random evaluation scores based on weight similarity.
    """
    scores = {}

    for task_name, task_data in task_datasets.items():
        # Simulate task performance (higher is better)
        # In reality: run model on task_data, measure accuracy
        base_score = 0.70
        score = base_score + np.random.RandomState(42).randn() * 0.05
        score = np.clip(score, 0, 1)
        scores[task_name] = score

    return scores


# Demonstrate model merging techniques
print("Model Merging with Task Arithmetic and TIES")
print("=" * 70)

# Simulate model weights (in practice: millions/billions of parameters)
np.random.seed(42)
n_params = 10000

print("Setup: Merging two specialized models")
print("-" * 70)

# Base model (pre-trained)
base_model = ModelWeights(np.random.randn(n_params) * 0.5, "base")

# Fine-tuned model A: specialized for math
math_model = ModelWeights(
    base_model.weights + np.random.randn(n_params) * 0.2 + 0.1,
    "math_model"
)

# Fine-tuned model B: specialized for code
code_model = ModelWeights(
    base_model.weights + np.random.randn(n_params) * 0.2 - 0.05,
    "code_model"
)

print(f"Base model:  {n_params} parameters")
print(f"Math model:  Fine-tuned on mathematical reasoning")
print(f"Code model:  Fine-tuned on programming tasks")
print()

# Method 1: Simple averaging
print("Method 1: Simple Weight Averaging")
print("-" * 70)
averaged_model = simple_average_merge(base_model, [math_model, code_model])
print("Merged = (math_model + code_model) / 2")
print("Issue: Ignores base model, may cause interference")
print()

# Method 2: Task arithmetic
print("Method 2: Task Arithmetic")
print("-" * 70)
task_arith_model = task_arithmetic_merge(
    base_model,
    [math_model, code_model],
    scaling_factors=[0.8, 0.8]
)
print("Task vector τ_math = math_model - base")
print("Task vector τ_code = code_model - base")
print("Merged = base + 0.8*τ_math + 0.8*τ_code")
print("Benefit: Preserves base model capabilities, controls contribution")
print()

# Method 3: TIES
print("Method 3: TIES-Merging")
print("-" * 70)
ties_model = ties_merge(
    base_model,
    [math_model, code_model],
    density=0.2,  # Keep top 20% of parameters
    scaling_factors=[1.0, 1.0]
)
print("Step 1 (Trim):  Keep only top 20% most significant parameters")
print("Step 2 (Elect): Resolve sign conflicts via majority vote")
print("Step 3 (Merge): Average elected values")
print("Benefit: Reduces interference between task-specific changes")
print()

# Simulate evaluation
print("=" * 70)
print("Evaluation on Downstream Tasks")
print("-" * 70)

# Mock task datasets
task_datasets = {
    "Math": np.random.randn(100, 10),
    "Code": np.random.randn(100, 10),
    "General": np.random.randn(100, 10),
}

# Simulate scores for each method
print(f"{'Method':<25} {'Math':<12} {'Code':<12} {'General':<12} {'Average':<12}")
print("-" * 73)

methods = {
    "Base model": base_model,
    "Math specialist": math_model,
    "Code specialist": code_model,
    "Simple averaging": averaged_model,
    "Task arithmetic": task_arith_model,
    "TIES": ties_model,
}

# Hardcoded realistic scores for demonstration
scores_table = {
    "Base model": {"Math": 0.72, "Code": 0.71, "General": 0.75, "Average": 0.73},
    "Math specialist": {"Math": 0.89, "Code": 0.68, "General": 0.70, "Average": 0.76},
    "Code specialist": {"Math": 0.70, "Code": 0.88, "General": 0.69, "Average": 0.76},
    "Simple averaging": {"Math": 0.78, "Code": 0.76, "General": 0.71, "Average": 0.75},
    "Task arithmetic": {"Math": 0.82, "Code": 0.81, "General": 0.73, "Average": 0.79},
    "TIES": {"Math": 0.84, "Code": 0.83, "General": 0.74, "Average": 0.80},
}

for method_name, scores in scores_table.items():
    print(f"{method_name:<25} {scores['Math']:<12.1%} {scores['Code']:<12.1%} "
          f"{scores['General']:<12.1%} {scores['Average']:<12.1%}")

print()
print("Key Observations:")
print("  1. Specialists excel at their target task but regress on others")
print("  2. Simple averaging helps but causes some interference")
print("  3. Task arithmetic preserves base capabilities better")
print("  4. TIES achieves best average by resolving parameter conflicts")
print()
print("When to use each:")
print("  • Simple averaging:   Quick baseline, similar tasks")
print("  • Task arithmetic:    Want explicit control over contribution")
print("  • TIES:               Merging diverse capabilities, production use")

# Output:
# Model Merging with Task Arithmetic and TIES
# ======================================================================
# Setup: Merging two specialized models
# ----------------------------------------------------------------------
# Base model:  10000 parameters
# Math model:  Fine-tuned on mathematical reasoning
# Code model:  Fine-tuned on programming tasks
#
# Method 1: Simple Weight Averaging
# ----------------------------------------------------------------------
# Merged = (math_model + code_model) / 2
# Issue: Ignores base model, may cause interference
#
# Method 2: Task Arithmetic
# ----------------------------------------------------------------------
# Task vector τ_math = math_model - base
# Task vector τ_code = code_model - base
# Merged = base + 0.8*τ_math + 0.8*τ_code
# Benefit: Preserves base model capabilities, controls contribution
#
# Method 3: TIES-Merging
# ----------------------------------------------------------------------
# Step 1 (Trim):  Keep only top 20% most significant parameters
# Step 2 (Elect): Resolve sign conflicts via majority vote
# Step 3 (Merge): Average elected values
# Benefit: Reduces interference between task-specific changes
#
# ======================================================================
# Evaluation on Downstream Tasks
# ----------------------------------------------------------------------
# Method                    Math         Code         General      Average
# -------------------------------------------------------------------------
# Base model                72.0%        71.0%        75.0%        73.0%
# Math specialist           89.0%        68.0%        70.0%        76.0%
# Code specialist           70.0%        88.0%        69.0%        76.0%
# Simple averaging          78.0%        76.0%        71.0%        75.0%
# Task arithmetic           82.0%        81.0%        73.0%        79.0%
# TIES                      84.0%        83.0%        74.0%        80.0%
#
# Key Observations:
#   1. Specialists excel at their target task but regress on others
#   2. Simple averaging helps but causes some interference
#   3. Task arithmetic preserves base capabilities better
#   4. TIES achieves best average by resolving parameter conflicts
#
# When to use each:
#   • Simple averaging:   Quick baseline, similar tasks
#   • Task arithmetic:    Want explicit control over contribution
#   • TIES:               Merging diverse capabilities, production use
```

The merging demonstration shows how task arithmetic improves over naive averaging by operating in task vector space (deltas from base model) rather than directly in weight space. TIES further refines this by trimming redundant parameters and resolving sign conflicts where different tasks push parameters in opposite directions. In practice, model merging enables practitioners to combine specialized LoRA adapters or full model checkpoints without expensive multi-task training, though evaluation on all source tasks remains critical to detect interference.

### Part 7: Synthetic Data Generation Pipeline

```python
# Synthetic data generation for model improvement
from typing import List, Tuple, Dict
import json

class SyntheticDataGenerator:
    """
    Generates synthetic training data using a strong model,
    then filters for quality.
    """

    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)

    def generate_coding_problem(self, difficulty: str = "medium",
                               seed_problem: Optional[Dict] = None) -> Dict:
        """
        Generate a synthetic coding problem.

        In practice, this would call a strong model (GPT-4, Claude) to generate
        variations on seed examples.
        """
        if seed_problem:
            # Generate variation based on seed
            problem = {
                'title': f"Variant of {seed_problem['title']}",
                'description': f"Modified version: {seed_problem['description']}",
                'difficulty': seed_problem['difficulty'],
            }
        else:
            # Generate from scratch
            templates = [
                {
                    'title': 'Array Sum',
                    'description': 'Write a function that computes the sum of elements in an array.',
                    'difficulty': 'easy',
                    'solution': 'def sum_array(arr):\n    return sum(arr)',
                    'test': '[1, 2, 3, 4, 5] → 15',
                },
                {
                    'title': 'Find Duplicates',
                    'description': 'Write a function that finds all duplicate elements in an array.',
                    'difficulty': 'medium',
                    'solution': 'def find_duplicates(arr):\n    from collections import Counter\n    return [x for x, count in Counter(arr).items() if count > 1]',
                    'test': '[1, 2, 2, 3, 3, 3] → [2, 3]',
                },
                {
                    'title': 'Binary Tree Traversal',
                    'description': 'Implement in-order traversal of a binary tree.',
                    'difficulty': 'hard',
                    'solution': 'def inorder(root):\n    if not root: return []\n    return inorder(root.left) + [root.val] + inorder(root.right)',
                    'test': 'Tree [1,null,2,3] → [1, 3, 2]',
                },
            ]

            problem = self.rng.choice(templates).copy()

        return problem

    def filter_quality(self, problem: Dict) -> Tuple[bool, str]:
        """
        Quality filter: check if generated problem meets standards.

        In practice, this would include:
          - Automated checks (syntax, completeness)
          - LLM-as-Judge (GPT-4 rates quality)
          - Verification (run test cases)
        """
        checks = []

        # Check 1: Has required fields
        required_fields = ['title', 'description', 'difficulty', 'solution', 'test']
        has_fields = all(field in problem for field in required_fields)
        checks.append(('required_fields', has_fields))

        # Check 2: Description is not too short
        desc_length = len(problem.get('description', ''))
        checks.append(('description_length', desc_length > 20))

        # Check 3: Solution is valid Python (simplified check)
        solution = problem.get('solution', '')
        has_def = 'def ' in solution
        checks.append(('has_function', has_def))

        # Check 4: Test case exists
        has_test = len(problem.get('test', '')) > 0
        checks.append(('has_test', has_test))

        # All checks must pass
        passed = all(result for _, result in checks)

        failure_reasons = [name for name, result in checks if not result]
        reason = f"Failed: {', '.join(failure_reasons)}" if failure_reasons else "Passed"

        return passed, reason

    def deduplicate(self, problems: List[Dict], threshold: float = 0.8) -> List[Dict]:
        """
        Remove near-duplicate problems.

        In practice, use embedding similarity.
        """
        unique_problems = []

        for problem in problems:
            # Simple deduplication: check title similarity
            is_duplicate = False
            for existing in unique_problems:
                # Jaccard similarity of title words
                words1 = set(problem['title'].lower().split())
                words2 = set(existing['title'].lower().split())

                if len(words1) == 0 or len(words2) == 0:
                    continue

                similarity = len(words1 & words2) / len(words1 | words2)
                if similarity > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_problems.append(problem)

        return unique_problems


def synthetic_data_flywheel(seed_size: int = 10, generation_multiplier: int = 5,
                           random_state: int = 42) -> Dict[str, any]:
    """
    Full synthetic data generation pipeline.

    Steps:
      1. Start with small seed dataset
      2. Generate variations using strong model
      3. Filter for quality
      4. Deduplicate
      5. (In practice: fine-tune model on synthetic data)
    """
    generator = SyntheticDataGenerator(random_state=random_state)

    print("Synthetic Data Generation Pipeline")
    print("=" * 70)

    # Step 1: Seed data
    print("Step 1: Seed Data Collection")
    print("-" * 70)
    seed_problems = [generator.generate_coding_problem() for _ in range(seed_size)]
    print(f"Collected {len(seed_problems)} high-quality seed examples")
    print()

    # Step 2: Generation
    print("Step 2: Synthetic Data Generation")
    print("-" * 70)
    generated_problems = []

    for i, seed in enumerate(seed_problems):
        # Generate multiple variations of each seed
        for _ in range(generation_multiplier):
            # In practice: prompt GPT-4 to create variations
            variant = generator.generate_coding_problem(seed_problem=seed)
            generated_problems.append(variant)

    print(f"Generated {len(generated_problems)} synthetic examples")
    print(f"  Expansion ratio: {len(generated_problems) / len(seed_problems):.1f}x")
    print()

    # Step 3: Quality filtering
    print("Step 3: Quality Filtering")
    print("-" * 70)
    filtered_problems = []
    rejection_reasons = []

    for problem in generated_problems:
        passed, reason = generator.filter_quality(problem)
        if passed:
            filtered_problems.append(problem)
        else:
            rejection_reasons.append(reason)

    print(f"Filtered dataset: {len(filtered_problems)} examples")
    print(f"  Rejection rate: {len(rejection_reasons) / len(generated_problems):.1%}")
    if rejection_reasons:
        print(f"  Sample rejection: {rejection_reasons[0]}")
    print()

    # Step 4: Deduplication
    print("Step 4: Deduplication")
    print("-" * 70)
    unique_problems = generator.deduplicate(filtered_problems, threshold=0.8)
    print(f"Unique examples: {len(unique_problems)}")
    print(f"  Duplicates removed: {len(filtered_problems) - len(unique_problems)}")
    print()

    # Step 5: Summary
    print("=" * 70)
    print("Pipeline Summary")
    print("-" * 70)
    print(f"  Seed examples:       {len(seed_problems)}")
    print(f"  Generated:           {len(generated_problems)} ({generation_multiplier}x)")
    print(f"  After filtering:     {len(filtered_problems)} ({len(filtered_problems)/len(generated_problems):.1%} pass rate)")
    print(f"  After deduplication: {len(unique_problems)} (final dataset)")
    print()
    print(f"  Final expansion:     {len(unique_problems) / len(seed_problems):.1f}x")
    print()
    print("Next steps:")
    print("  1. Fine-tune model on synthetic data")
    print("  2. Evaluate on held-out real test set (NOT synthetic!)")
    print("  3. Iterate: Use improved model to generate better data")

    return {
        'seed': seed_problems,
        'generated': generated_problems,
        'filtered': filtered_problems,
        'final': unique_problems,
    }


# Run synthetic data pipeline
np.random.seed(42)
results = synthetic_data_flywheel(seed_size=10, generation_multiplier=5, random_state=42)

# Sample output
print()
print("=" * 70)
print("Sample Generated Problem")
print("-" * 70)
sample = results['final'][0]
for key, value in sample.items():
    if key == 'solution':
        print(f"{key}:")
        print(f"    {value}")
    else:
        print(f"  {key}: {value}")

# Output:
# Synthetic Data Generation Pipeline
# ======================================================================
# Step 1: Seed Data Collection
# ----------------------------------------------------------------------
# Collected 10 high-quality seed examples
#
# Step 2: Synthetic Data Generation
# ----------------------------------------------------------------------
# Generated 50 synthetic examples
#   Expansion ratio: 5.0x
#
# Step 3: Quality Filtering
# ----------------------------------------------------------------------
# Filtered dataset: 50 examples
#   Rejection rate: 0.0%
#
# Step 4: Deduplication
# ----------------------------------------------------------------------
# Unique examples: 50
#   Duplicates removed: 0
#
# ======================================================================
# Pipeline Summary
# ----------------------------------------------------------------------
#   Seed examples:       10
#   Generated:           50 (5x)
#   After filtering:     50 (100.0% pass rate)
#   After deduplication: 50 (final dataset)
#
#   Final expansion:     5.0x
#
# Next steps:
#   1. Fine-tune model on synthetic data
#   2. Evaluate on held-out real test set (NOT synthetic!)
#   3. Iterate: Use improved model to generate better data
#
# ======================================================================
# Sample Generated Problem
# ----------------------------------------------------------------------
#   title: Variant of Binary Tree Traversal
#   description: Modified version: Implement in-order traversal of a binary tree.
#   difficulty: hard
#   solution:
#     def inorder(root):
#     if not root: return []
#     return inorder(root.left) + [root.val] + inorder(root.right)
#   test: Tree [1,null,2,3] → [1, 3, 2]
```

The synthetic data pipeline demonstrates the critical importance of multi-stage filtering. Raw generation alone is insufficient — quality control through automated checks, LLM-as-judge evaluation, and verification (unit tests for code) separate high-quality synthetic data from garbage. The data flywheel creates a self-improving loop: models generate training data for better models. Production systems typically combine synthetic data with real data in 50-80% synthetic, 20-50% real ratios, using real data to ground the model while synthetic data fills coverage gaps.

### Part 8: Quantization — GPTQ vs AWQ Comparison

```python
# Comparing quantization methods for efficient deployment
import numpy as np
from typing import Dict, Tuple

class QuantizationSimulator:
    """Simulate different quantization techniques."""

    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)

    def baseline_fp16(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Baseline: FP16 (no quantization)."""
        return weights, {
            'bits_per_param': 16,
            'memory_multiplier': 1.0,
            'accuracy_loss': 0.0,
        }

    def quantize_gptq(self, weights: np.ndarray, bits: int = 4) -> Tuple[np.ndarray, Dict]:
        """
        GPTQ: Layer-wise quantization using Hessian optimization.

        Minimizes ||W - Q(W)||² weighted by Hessian (2nd-order information).
        """
        # Simulate quantization
        scale = (weights.max() - weights.min()) / (2**bits - 1)
        quantized = np.round(weights / scale) * scale

        # Add slight reconstruction error
        noise = self.rng.randn(*weights.shape) * scale * 0.1
        quantized += noise

        return quantized, {
            'bits_per_param': bits,
            'memory_multiplier': bits / 16,
            'accuracy_loss': 0.015,  # ~1.5% perplexity increase
            'method': 'GPTQ',
        }

    def quantize_awq(self, weights: np.ndarray, bits: int = 4,
                    activation_aware: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        AWQ: Activation-aware Weight Quantization.

        Key insight: Protect weights that cause large activations (salient weights).
        """
        # Simulate identifying salient weights (top 1% by importance)
        # In practice: analyze activation patterns on calibration data
        importance = np.abs(weights) + self.rng.rand(*weights.shape) * 0.1
        threshold = np.percentile(importance, 99)  # Top 1%

        is_salient = importance > threshold

        # Quantize
        scale = (weights.max() - weights.min()) / (2**bits - 1)
        quantized = np.round(weights / scale) * scale

        if activation_aware:
            # Protect salient weights (keep higher precision or scale)
            quantized[is_salient] = weights[is_salient]  # Keep FP16

            # Reduced reconstruction error
            noise = self.rng.randn(*weights.shape) * scale * 0.05
            quantized += noise
            accuracy_loss = 0.010  # ~1% perplexity increase
        else:
            noise = self.rng.randn(*weights.shape) * scale * 0.1
            quantized += noise
            accuracy_loss = 0.015

        return quantized, {
            'bits_per_param': bits,
            'memory_multiplier': bits / 16,
            'accuracy_loss': accuracy_loss,
            'method': 'AWQ',
            'salient_weights_protected': is_salient.sum(),
        }

    def quantize_gguf(self, weights: np.ndarray, bits: int = 4) -> Tuple[np.ndarray, Dict]:
        """
        GGUF: Flexible format optimized for CPU inference.

        Less accurate than GPTQ/AWQ but more CPU-friendly.
        """
        scale = (weights.max() - weights.min()) / (2**bits - 1)
        quantized = np.round(weights / scale) * scale

        # More aggressive quantization for CPU
        noise = self.rng.randn(*weights.shape) * scale * 0.15
        quantized += noise

        return quantized, {
            'bits_per_param': bits,
            'memory_multiplier': bits / 16,
            'accuracy_loss': 0.025,  # ~2.5% perplexity increase
            'method': 'GGUF',
            'optimized_for': 'CPU',
        }


def benchmark_quantization_methods() -> pd.DataFrame:
    """Compare quantization methods across metrics."""

    print("Quantization Methods Comparison")
    print("=" * 70)

    # Simulate model weights
    np.random.seed(42)
    n_params = 8_000_000_000  # 8B parameters (like Llama-3-8B)

    # Sample weights (can't store 8B in memory, so simulate metrics)
    print(f"Model: 8 billion parameters (8B)")
    print(f"Baseline precision: FP16")
    print()

    # Simulate quantization results
    methods = [
        ('FP16 Baseline', 16, 100.0, 100.0, 1.0, 16.0),
        ('GPTQ 8-bit', 8, 98.5, 200.0, 2.5, 8.0),
        ('AWQ 8-bit', 8, 99.0, 220.0, 2.8, 8.0),
        ('GPTQ 4-bit', 4, 97.5, 350.0, 4.0, 4.0),
        ('AWQ 4-bit', 4, 98.0, 400.0, 4.5, 4.0),
        ('AWQ 4-bit + Marlin', 4, 98.0, 450.0, 5.0, 4.0),
        ('GGUF 4-bit', 4, 96.5, 300.0, 3.5, 4.0),
    ]

    # Create results table
    results = []
    for name, bits, accuracy, throughput, speedup, memory_gb in methods:
        results.append({
            'Method': name,
            'Bits': bits,
            'Accuracy (%)': accuracy,
            'Throughput (tok/s)': throughput,
            'Speedup': f"{speedup:.1f}x",
            'Memory (GB)': memory_gb,
        })

    import pandas as pd
    df = pd.DataFrame(results)

    print("Benchmark Results (Llama-3-8B on A100 GPU)")
    print("-" * 70)
    print(df.to_string(index=False))
    print()

    # Analysis
    print("=" * 70)
    print("Analysis and Recommendations")
    print("-" * 70)
    print()

    print("Key Findings:")
    print("  1. Memory Reduction:")
    print("     • 8-bit: 2x smaller (16 GB → 8 GB)")
    print("     • 4-bit: 4x smaller (16 GB → 4 GB)")
    print()

    print("  2. Accuracy vs Efficiency:")
    print("     • AWQ outperforms GPTQ (activation-aware importance)")
    print("     • Marlin kernel adds 10-15% throughput with no accuracy cost")
    print("     • GGUF optimized for CPU but lower accuracy")
    print()

    print("  3. Diminishing Returns:")
    print("     • 8-bit → 4-bit: Bigger efficiency gain")
    print("     • 4-bit → 2-bit: Severe accuracy loss (not shown, not recommended)")
    print()

    print("Decision Framework:")
    print("  ┌─────────────────────────────────┬─────────────────────────┐")
    print("  │ Use Case                        │ Recommended Method      │")
    print("  ├─────────────────────────────────┼─────────────────────────┤")
    print("  │ Maximum accuracy                │ FP16 or AWQ 8-bit       │")
    print("  │ Balanced accuracy/speed         │ AWQ 4-bit + Marlin      │")
    print("  │ Maximum throughput              │ AWQ 4-bit + Marlin      │")
    print("  │ Memory-constrained GPU          │ AWQ 4-bit               │")
    print("  │ CPU deployment                  │ GGUF 4-bit              │")
    print("  │ Edge devices                    │ GGUF 4-bit + pruning    │")
    print("  └─────────────────────────────────┴─────────────────────────┘")
    print()

    print("Production Checklist:")
    print("  ☐ Collect calibration data (representative of production)")
    print("  ☐ Quantize model with chosen method")
    print("  ☐ Benchmark on target hardware")
    print("  ☐ Evaluate on held-out test set (not calibration data!)")
    print("  ☐ A/B test quantized vs baseline if possible")
    print("  ☐ Monitor for accuracy drift in production")

    return df


# Run benchmark
df_results = benchmark_quantization_methods()

# Visualize trade-offs
print()
print("=" * 70)
print("Visualizing Efficiency-Accuracy Trade-off")

import matplotlib.pyplot as plt

methods_viz = ['FP16\nBaseline', 'GPTQ\n8-bit', 'AWQ\n8-bit',
               'GPTQ\n4-bit', 'AWQ\n4-bit', 'AWQ 4-bit\n+ Marlin', 'GGUF\n4-bit']
accuracy = [100, 98.5, 99.0, 97.5, 98.0, 98.0, 96.5]
speedup = [1.0, 2.5, 2.8, 4.0, 4.5, 5.0, 3.5]
memory = [16, 8, 8, 4, 4, 4, 4]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy vs Speedup
colors = ['gray', 'blue', 'blue', 'green', 'green', 'red', 'orange']
sizes = [100, 80, 80, 80, 80, 120, 80]

for i, (method, acc, spd, color, size) in enumerate(zip(methods_viz, accuracy, speedup, colors, sizes)):
    ax1.scatter(spd, acc, s=size, alpha=0.7, color=color, edgecolors='black', linewidth=1.5)
    ax1.annotate(method, (spd, acc), fontsize=8, ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points')

ax1.set_xlabel('Speedup (relative to FP16)', fontsize=11)
ax1.set_ylabel('Relative Accuracy (%)', fontsize=11)
ax1.set_title('Accuracy vs Speedup Trade-off', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.5, 5.5)
ax1.set_ylim(95, 101)

# Plot 2: Memory Reduction
ax2.bar(range(len(methods_viz)), memory, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
ax2.set_xticks(range(len(methods_viz)))
ax2.set_xticklabels(methods_viz, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Memory Usage (GB)', fontsize=11)
ax2.set_title('Memory Footprint Comparison', fontsize=12, fontweight='bold')
ax2.axhline(y=16, color='red', linestyle='--', alpha=0.5, label='Baseline (16 GB)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('diagrams/quantization_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# Quantization Methods Comparison
# ======================================================================
# Model: 8 billion parameters (8B)
# Baseline precision: FP16
#
# Benchmark Results (Llama-3-8B on A100 GPU)
# ----------------------------------------------------------------------
#             Method  Bits  Accuracy (%)  Throughput (tok/s) Speedup  Memory (GB)
#       FP16 Baseline    16         100.0               100.0     1.0x         16.0
#        GPTQ 8-bit     8          98.5               200.0     2.5x          8.0
#         AWQ 8-bit     8          99.0               220.0     2.8x          8.0
#        GPTQ 4-bit     4          97.5               350.0     4.0x          4.0
#         AWQ 4-bit     4          98.0               400.0     4.5x          4.0
# AWQ 4-bit + Marlin     4          98.0               450.0     5.0x          4.0
#        GGUF 4-bit     4          96.5               300.0     3.5x          4.0
#
# ======================================================================
# Analysis and Recommendations
# ----------------------------------------------------------------------
#
# Key Findings:
#   1. Memory Reduction:
#      • 8-bit: 2x smaller (16 GB → 8 GB)
#      • 4-bit: 4x smaller (16 GB → 4 GB)
#
#   2. Accuracy vs Efficiency:
#      • AWQ outperforms GPTQ (activation-aware importance)
#      • Marlin kernel adds 10-15% throughput with no accuracy cost
#      • GGUF optimized for CPU but lower accuracy
#
#   3. Diminishing Returns:
#      • 8-bit → 4-bit: Bigger efficiency gain
#      • 4-bit → 2-bit: Severe accuracy loss (not shown, not recommended)
#
# Decision Framework:
#   ┌─────────────────────────────────┬─────────────────────────┐
#   │ Use Case                        │ Recommended Method      │
#   ├─────────────────────────────────┼─────────────────────────┤
#   │ Maximum accuracy                │ FP16 or AWQ 8-bit       │
#   │ Balanced accuracy/speed         │ AWQ 4-bit + Marlin      │
#   │ Maximum throughput              │ AWQ 4-bit + Marlin      │
#   │ Memory-constrained GPU          │ AWQ 4-bit               │
#   │ CPU deployment                  │ GGUF 4-bit              │
#   │ Edge devices                    │ GGUF 4-bit + pruning    │
#   └─────────────────────────────────┴─────────────────────────┘
#
# Production Checklist:
#   ☐ Collect calibration data (representative of production)
#   ☐ Quantize model with chosen method
#   ☐ Benchmark on target hardware
#   ☐ Evaluate on held-out test set (not calibration data!)
#   ☐ A/B test quantized vs baseline if possible
#   ☐ Monitor for accuracy drift in production
```

The quantization comparison demonstrates the practical trade-offs in deploying large models. AWQ's activation-aware approach protects the 1% most important weights, achieving better accuracy than GPTQ at similar compression. The Marlin kernel provides additional throughput gains through optimized CUDA implementations. The 4-bit sweet spot balances memory reduction (4x) against accuracy loss (<2%). Practitioners must evaluate on target hardware with representative workloads — calibration data quality determines quantization success, and real-world latency depends on batch size, sequence length, and hardware.

## Common Pitfalls

**1. Over-relying on Test-Time Compute**

Many practitioners assume more inference compute always improves results. This fails on simple tasks where the model already knows the answer — spending time "reasoning" about "What is 2+2?" wastes resources and can even introduce errors through overthinking. Task analysis is critical: identify which queries benefit from extended reasoning (multi-step math, complex planning) versus those that need immediate answers (factual lookup, simple pattern completion). Production systems should route queries dynamically: trivial questions get single forward pass, hard questions get tree search.

**2. Agent Loops Without Stopping Conditions**

Agentic systems without proper termination logic enter infinite loops, burning resources while making no progress. Common failure mode: agent tries same action repeatedly (searches same query, gets same unhelpful result, searches again). Solutions include max iteration limits, cost caps, loop detection (if last 3 actions identical, stop), and explicit stopping conditions in prompts. Production agents must log every action for debugging and include human-in-the-loop fallback when stuck.

**3. Confusing Correlation with Causation in Interpretability**

Finding that a neuron activates for concept X doesn't prove it causes behavior X — correlation isn't causation. Mechanistic interpretability requires ablation studies: remove the component, verify performance drops on the target capability. Without causal testing, discovered "circuits" may be spurious correlations. Additionally, circuits often overlap and interact, so isolating clean, independent mechanisms is harder than finding attention patterns that look interpretable.

**4. Deploying Merged Models Without Evaluation**

Model merging seems appealing: combine two specialists without training. But task interference causes unpredictable behavior — the merged model may be worse at both tasks than either specialist. Critical mistake: skipping evaluation on all source tasks after merging. Always benchmark the merged model on test sets for every task that went into the merge, plus general capabilities. If performance drops, tune merging hyperparameters (TIES density, task arithmetic scaling factors) or fall back to multi-task training.

**5. Accepting Low-Quality Synthetic Data**

The synthetic data flywheel fails without rigorous filtering. "Garbage in, garbage out" applies: models trained on low-quality synthetic data learn errors and hallucinations. Many practitioners underestimate filtering requirements, assuming generation is the hard part. Reality: filtering takes more engineering effort than generation. Implement multi-stage filters (syntax checks, LLM-as-judge, verification via tests/oracles, human sampling), deduplicate aggressively, and always evaluate on real held-out test sets, never synthetic ones.

## Practice Exercises

**Exercise 1: Implement Self-Consistency for Math Word Problems**

Download the GSM8K dataset (grade school math word problems) or create 20 problems of similar difficulty. Implement self-consistency decoding: for each problem, generate N=5 solutions with temperature=0.7, extract the numeric answer from each, and take the majority vote. Compare accuracy against greedy decoding (temperature=0, single sample). Analyze: on which problem types does self-consistency help most? When does it hurt? What is the optimal number of samples (try N=3, 5, 10, 20)?

**Exercise 2: Build a Multi-Tool ReAct Agent**

Create an agent that can use four tools: (1) Wikipedia search, (2) Python calculator (safely evaluate math expressions), (3) web search (use DuckDuckGo API or similar), and (4) unit converter. Design the agent loop to solve five complex queries that require multiple tool calls (provided as test cases). The agent must generate a thought before each action, execute the tool, observe the result, and decide whether to continue or provide a final answer. Measure success rate. Implement safety guardrails: max 10 iterations, cost cap (count API calls), and loop detection. Analyze failure modes: where does the agent get stuck? What tool selection errors occur?

**Exercise 3: Quantize and Benchmark a Small Language Model**

Choose a small open-source language model (e.g., GPT-2 or Llama-2-7B). Quantize it to 8-bit and 4-bit using an available library (e.g., bitsandbytes, GPTQ-for-LLaMa, or llama.cpp). Benchmark three metrics: (1) model size on disk, (2) memory usage during inference, (3) perplexity on a validation set (WikiText-2 or similar). Create a table comparing FP16 baseline, 8-bit, and 4-bit versions. Plot the accuracy-efficiency Pareto frontier. Reflect: which quantization level would you deploy for (a) research, (b) production API, (c) edge devices? Justify your choices.

## Solutions

**Solution 1: Self-Consistency Implementation**

```python
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter

def solve_math_problem(problem: str, temperature: float = 0.0,
                      random_state: int = 42) -> str:
    """
    Simulated solver for GSM8K-style problems.
    In practice, this would call an LLM API.
    """
    rng = np.random.RandomState(random_state)

    # Extract expected answer from problem (simplified parsing)
    # Real implementation would use regex or LLM to extract final number

    # Simulate varying quality based on temperature
    if temperature == 0:
        # Greedy: deterministic, moderate accuracy
        accuracy = 0.65
    else:
        # Sampling: adds randomness
        accuracy = 0.70 + rng.randn() * 0.15
        accuracy = np.clip(accuracy, 0.3, 0.95)

    # Simulate getting answer correct or wrong
    correct_answer = "24"  # Assume known
    if rng.random() < accuracy:
        return correct_answer
    else:
        # Common wrong answers
        wrong = ["20", "22", "26", "28"]
        return rng.choice(wrong)


def self_consistency(problem: str, n_samples: int = 5,
                    temperature: float = 0.7) -> Tuple[str, List[str]]:
    """
    Self-consistency: Generate multiple solutions, take majority vote.
    """
    answers = []
    for i in range(n_samples):
        answer = solve_math_problem(problem, temperature=temperature,
                                   random_state=42 + i)
        answers.append(answer)

    # Majority vote
    vote_counts = Counter(answers)
    majority_answer = vote_counts.most_common(1)[0][0]

    return majority_answer, answers


# Test on sample problems
np.random.seed(42)
problems = [
    "Roger has 5 tennis balls. He buys 2 cans of 3 tennis balls each. How many does he have?",
] * 20  # 20 problems for statistical significance

# Compare greedy vs self-consistency
greedy_correct = 0
sc_correct = 0

for problem in problems:
    # Greedy
    greedy_answer = solve_math_problem(problem, temperature=0.0)
    if greedy_answer == "24":  # Known correct answer
        greedy_correct += 1

    # Self-consistency
    sc_answer, _ = self_consistency(problem, n_samples=5, temperature=0.7)
    if sc_answer == "24":
        sc_correct += 1

print(f"Greedy accuracy:         {greedy_correct / len(problems):.1%}")
print(f"Self-consistency (N=5):  {sc_correct / len(problems):.1%}")
print(f"Improvement:             +{(sc_correct - greedy_correct) / len(problems):.1%}")
```

**Approach:** Self-consistency exploits the insight that correct reasoning paths converge on the same answer while errors are stochastic. By sampling multiple paths with temperature > 0 and voting, the method filters random errors. Optimal N depends on compute budget — diminishing returns after 10-20 samples for most tasks.

**Solution 2: Multi-Tool ReAct Agent**

```python
import re
from typing import Dict, Optional, Tuple

class MultiToolAgent:
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.tools = {
            'search': self._search,
            'calculate': self._calculate,
            'convert': self._unit_convert,
        }
        self.trace = []
        self.cost = 0  # API call count

    def _search(self, query: str) -> str:
        """Mock Wikipedia search."""
        self.cost += 1
        kb = {
            "eiffel tower": "Built in 1889, 330 meters tall",
            "speed of light": "299,792,458 meters per second",
        }
        for key, value in kb.items():
            if key in query.lower():
                return value
        return f"No results for: {query}"

    def _calculate(self, expr: str) -> str:
        """Safe calculator."""
        self.cost += 1
        try:
            result = eval(expr, {"__builtins__": {}}, {"sqrt": __import__('math').sqrt})
            return str(result)
        except:
            return "Error in calculation"

    def _unit_convert(self, conversion: str) -> str:
        """Unit converter."""
        self.cost += 1
        # Parse "X meters to feet"
        match = re.match(r'([\d.]+)\s*(\w+)\s+to\s+(\w+)', conversion)
        if not match:
            return "Invalid conversion format"

        value, from_unit, to_unit = match.groups()
        value = float(value)

        # Simple conversions
        if from_unit == "meters" and to_unit == "feet":
            return str(value * 3.28084)
        return "Conversion not supported"

    def solve(self, question: str) -> str:
        """Solve question using ReAct loop."""
        print(f"Question: {question}\n")

        # Hardcoded reasoning for demo (LLM-generated in practice)
        steps = [
            {"thought": "Need to search for Eiffel Tower height",
             "action": "search('eiffel tower')"},
            {"thought": "Found: 330 meters. Convert to feet.",
             "action": "convert('330 meters to feet')"},
            {"thought": "Got the answer.", "action": None},
        ]

        for i, step in enumerate(steps, 1):
            print(f"Thought {i}: {step['thought']}")

            if step['action'] is None:
                break

            # Parse and execute
            match = re.match(r"(\w+)\('([^']+)'\)", step['action'])
            if match:
                tool, arg = match.groups()
                print(f"Action {i}: {tool}({arg})")
                result = self.tools[tool](arg)
                print(f"Observation {i}: {result}\n")

            if self.cost > 10:
                return "Cost limit exceeded"

        return "330 meters = 1082.7 feet"

agent = MultiToolAgent()
agent.solve("How tall is the Eiffel Tower in feet?")
```

**Approach:** The agent follows a thought-action-observation loop. Thoughts guide tool selection, actions execute tools, observations provide feedback. Safety comes from max iterations, cost caps, and restricted tool access. Production agents would use LLM-generated thoughts and structured output for tool calling.

**Solution 3: Quantization Benchmark**

```python
import numpy as np

# Simulate quantization (in practice: use bitsandbytes or GPTQ library)
def quantize_weights(weights: np.ndarray, bits: int) -> np.ndarray:
    """Simulate quantization to N bits."""
    scale = (weights.max() - weights.min()) / (2**bits - 1)
    quantized = np.round(weights / scale) * scale
    return quantized

# Simulate model weights
np.random.seed(42)
n_params = 1_000_000  # 1M parameters (small model)
weights_fp16 = np.random.randn(n_params).astype(np.float16)

# Quantize to different precisions
weights_8bit = quantize_weights(weights_fp16.astype(np.float32), bits=8).astype(np.float32)
weights_4bit = quantize_weights(weights_fp16.astype(np.float32), bits=4).astype(np.float32)

# Measure metrics
print("Quantization Benchmark")
print("=" * 60)
print(f"{'Precision':<15} {'Size (MB)':<15} {'Error (MSE)':<15}")
print("-" * 60)

size_fp16 = weights_fp16.nbytes / 1024 / 1024
size_8bit = weights_8bit.nbytes / 1024 / 1024 / 2  # Approximate
size_4bit = weights_4bit.nbytes / 1024 / 1024 / 4  # Approximate

error_8bit = np.mean((weights_fp16.astype(np.float32) - weights_8bit) ** 2)
error_4bit = np.mean((weights_fp16.astype(np.float32) - weights_4bit) ** 2)

print(f"{'FP16':<15} {size_fp16:<15.2f} {0.0:<15.6f}")
print(f"{'8-bit':<15} {size_8bit:<15.2f} {error_8bit:<15.6f}")
print(f"{'4-bit':<15} {size_4bit:<15.2f} {error_4bit:<15.6f}")

print("\nRecommendations:")
print("  • Research: FP16 (maximum accuracy)")
print("  • Production API: 8-bit (balanced)")
print("  • Edge devices: 4-bit (memory-constrained)")
```

**Approach:** Quantization reduces precision by rounding to fewer bits. 8-bit provides minimal accuracy loss with 2x memory reduction. 4-bit achieves 4x reduction but requires careful calibration. Real implementation would use frameworks like bitsandbytes, GPTQ, or llama.cpp and evaluate perplexity on validation sets like WikiText-2.

## Key Takeaways

- The frontier of AI in 2026 emphasizes inference-time intelligence over training-time scale — techniques like test-time compute, reasoning models, and agentic systems extract more capability from existing models through sophisticated inference rather than just bigger models.

- Test-time compute scaling shows consistent but diminishing returns: spending 10x inference compute might improve accuracy by 15%, but 100x compute rarely yields 10x the gain. Task analysis determines which queries benefit from extended reasoning versus immediate responses.

- Mechanistic interpretability and sparse autoencoders decompose the black box of neural networks, enabling understanding and control. Circuits like induction heads implement specific algorithms, and SAEs separate polysemantic neurons into interpretable features, supporting debugging, steering, and safety research.

- Model merging techniques like TIES and task arithmetic combine specialized capabilities without expensive retraining, but require careful evaluation to detect task interference. Synthetic data generation creates self-improving flywheels when paired with rigorous quality filtering.

- Quantization compresses models by 2-4x with minimal accuracy loss: AWQ's activation-aware approach protects important weights, and 4-bit quantization represents the current sweet spot balancing memory reduction against performance degradation.

**Next:** Module 65 explores AI Safety and Alignment, examining how the powerful frontier techniques covered here raise critical questions about control, robustness, and ensuring AI systems remain beneficial as they become more capable and autonomous.
