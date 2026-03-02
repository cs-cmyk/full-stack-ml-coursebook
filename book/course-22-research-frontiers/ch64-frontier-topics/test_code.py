#!/usr/bin/env python3
"""
Test script to validate all code blocks from content.md
"""

import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

def test_block_1_visualization():
    """Test: Visualization block"""
    print("Testing Block 1: Frontier Overview Visualization...")
    try:
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
        plt.close()

        print("✓ Block 1 PASSED\n")
        return True
    except Exception as e:
        print(f"✗ Block 1 FAILED: {e}\n")
        traceback.print_exc()
        return False


def test_block_2_best_of_n():
    """Test: Best-of-N Sampling"""
    print("Testing Block 2: Test-Time Compute - Best-of-N Sampling...")
    try:
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

        print("✓ Block 2 PASSED\n")
        return True
    except Exception as e:
        print(f"✗ Block 2 FAILED: {e}\n")
        traceback.print_exc()
        return False


def test_block_3_react_agent():
    """Test: ReAct Agent"""
    print("Testing Block 3: Agentic AI - ReAct Agent...")
    try:
        from typing import Dict, List, Callable, Optional, Tuple
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

        print("✓ Block 3 PASSED\n")
        return True
    except Exception as e:
        print(f"✗ Block 3 FAILED: {e}\n")
        traceback.print_exc()
        return False


def test_block_4_chain_of_thought():
    """Test: Chain-of-Thought Prompting"""
    print("Testing Block 4: Chain-of-Thought Prompting Comparison...")
    try:
        import numpy as np
        from typing import List, Tuple
        from collections import Counter

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

        print("✓ Block 4 PASSED\n")
        return True
    except Exception as e:
        print(f"✗ Block 4 FAILED: {e}\n")
        traceback.print_exc()
        return False


# Continue with remaining test functions...
def run_all_tests():
    """Run all code block tests"""
    results = []

    # Run tests
    results.append(("Visualization", test_block_1_visualization()))
    results.append(("Best-of-N Sampling", test_block_2_best_of_n()))
    results.append(("ReAct Agent", test_block_3_react_agent()))
    results.append(("Chain-of-Thought", test_block_4_chain_of_thought()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:<30} {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    return all(p for _, p in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
