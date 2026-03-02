# Glossary: Chapter 36 - Reinforcement Learning

## Terms Introduced in Section 36.1: Markov Decision Processes (MDPs)

### Markov Decision Process (MDP)
A mathematical framework for sequential decision-making under the Markov assumption, defined by a 5-tuple (S, A, P, R, γ) representing states, actions, transition probabilities, rewards, and discount factor. MDPs provide the foundation for reinforcement learning by formalizing how agents make decisions in environments where actions have delayed consequences.

**First introduced:** Section 36.1
**Related terms:** Policy, Value Function, Bellman Equation

---

### Policy
A strategy that maps states to actions, defining an agent's behavior in an MDP. Can be:
- **Deterministic:** π(s) = a (always choose action a in state s)
- **Stochastic:** π(a|s) = probability of choosing action a in state s

**First introduced:** Section 36.1
**Notation:** π
**Related terms:** Optimal Policy, Value Function

---

### Value Function
The expected cumulative discounted reward starting from a state and following a policy. Formally, V^π(s) represents "how good is state s when following policy π?" The value function captures the long-term desirability of states, accounting for both immediate and future rewards.

**First introduced:** Section 36.1
**Notation:** V^π(s) for policy π, V*(s) for optimal value
**Related terms:** Action-Value Function, Bellman Equation

---

### Action-Value Function (Q-function)
The expected cumulative discounted reward starting from a state, taking a specific action, then following a policy. Formally, Q^π(s,a) represents "how good is taking action a in state s when following policy π thereafter?" The Q-function is central to many RL algorithms including Q-learning.

**First introduced:** Section 36.1
**Notation:** Q^π(s,a) for policy π, Q*(s,a) for optimal action-value
**Related terms:** Value Function, Q-Learning (Section 36.2)

---

### Bellman Equation
A recursive relationship expressing the value of a state as the immediate reward plus the discounted value of successor states. The Bellman equation is the foundation for dynamic programming in RL:

V^π(s) = E[R + γ V^π(s') | s, π]

Two key variants:
- **Bellman Expectation Equation:** For evaluating a given policy
- **Bellman Optimality Equation:** For finding the optimal policy

**First introduced:** Section 36.1
**Named after:** Richard Bellman (1920-1984)
**Related terms:** Value Iteration, Policy Evaluation, Dynamic Programming

---

### Discount Factor
A parameter γ ∈ [0,1] that determines how much future rewards are valued relative to immediate rewards in an MDP. Higher values (e.g., 0.99) create "patient" agents that plan far ahead; lower values (e.g., 0.5) create "myopic" agents that prioritize immediate rewards. Typical values in practice are 0.9-0.99.

**First introduced:** Section 36.1
**Notation:** γ (gamma)
**Typical range:** 0.9 to 0.99
**Related terms:** Value Function, Bellman Equation

---

### Policy Evaluation
An iterative algorithm that computes the value function V^π(s) for a given policy by repeatedly applying the Bellman expectation equation until convergence. Also called "prediction" in reinforcement learning. Policy evaluation answers: "How good is my current policy?"

**First introduced:** Section 36.1
**Algorithm type:** Dynamic Programming
**Related terms:** Value Iteration, Temporal Difference Learning (Section 36.2)

---

### Value Iteration
An iterative algorithm that finds the optimal value function V*(s) and optimal policy π*(s) by repeatedly applying the Bellman optimality equation. Combines policy evaluation and policy improvement in a single update. Value iteration directly searches for the best possible policy.

**First introduced:** Section 36.1
**Algorithm type:** Dynamic Programming
**Convergence:** Guaranteed for finite MDPs with appropriate discount factor
**Related terms:** Policy Iteration, Q-Learning (Section 36.2)

---

### Markov Property
The property that the future depends only on the current state, not on the history of past states. Formally: P(s_{t+1} | s_t, a_t, s_{t-1}, ..., s_0) = P(s_{t+1} | s_t, a_t). This "memoryless" property is what makes MDPs tractable - without it, the agent would need to remember and condition on the entire history, leading to exponential state spaces.

**First introduced:** Section 36.1
**Named after:** Andrey Markov (1856-1922)
**Related terms:** MDP, State Representation

---

### Terminal State
An absorbing state in an MDP from which no further transitions occur (e.g., goal states, game-over states, or trap states). Once a terminal state is reached, the episode ends. Terminal states have value V(s) = 0 by convention (no future rewards possible), and transition to themselves with probability 1.

**First introduced:** Section 36.1
**Also called:** Absorbing state
**Related terms:** Episodic Task, Goal State

---

## Notation Summary

| Symbol | Meaning | Introduced |
|--------|---------|-----------|
| γ (gamma) | Discount factor | Section 36.1 |
| π (pi) | Policy | Section 36.1 |
| V^π(s) | State-value function under policy π | Section 36.1 |
| V*(s) | Optimal state-value function | Section 36.1 |
| Q^π(s,a) | Action-value function under policy π | Section 36.1 |
| Q*(s,a) | Optimal action-value function | Section 36.1 |
| s, s' | State, next state | Section 36.1 |
| a | Action | Section 36.1 |
| r, R | Reward | Section 36.1 |
| P(s'|s,a) | Transition probability | Section 36.1 |

---

## Cross-References

Terms that will be expanded in later sections:
- **Q-Learning** → Section 36.2
- **SARSA** → Section 36.2
- **Temporal Difference Learning** → Section 36.2
- **Function Approximation** → Section 36.3
- **Deep Q-Networks (DQN)** → Section 36.3
- **Policy Gradients** → Section 36.4

---

*Note: This glossary will be updated as new sections are added to Chapter 36.*
