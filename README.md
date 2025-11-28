# Summary of "Comparative Analysis of AI Agents for Othello"

## Overview
This project compares 12 different AI agents for playing Othello, ranging from simple heuristics to advanced reinforcement learning approaches. The research evaluates their relative performance through a round-robin tournament to identify the most effective strategies.

## Key Findings

### Agent Performance Rankings
The tournament revealed a clear hierarchy of effectiveness:

1. **MCTS-MAB (Multi-Armed Bandit)** - Strongest performer with 100% win rate against non-MCTS opponents and 70% against MCTS variants
2. **MCTS** - Achieved 80-100% wins against most agents, except MCTS-MAB and AlphaBeta4
3. **AlphaBeta4** - Strong results (90-100%) against non-MCTS agents, but struggled with MCTS approaches

### Major Insights

**Search-Based Dominance**: Traditional search methods significantly outperformed reinforcement learning approaches in this evaluation. Enhanced exploration strategies (as in MCTS-MAB) proved particularly effective.

**Reinforcement Learning Challenges**: RL agents underperformed expectations due to limited training time, simple neural network architectures, and lack of hyperparameter tuning. We note this doesn't fairly represent RL potential.

**Surprising Results**: The simple Greedy approach outperformed all RL agents, while the Hybrid agent showed balanced but unremarkable performance.

## Methodology

**Agents Tested**:
- Traditional: Random, Greedy, AlphaBeta (depths 1,3,4), MCTS
- Reinforcement Learning: Q-Learning, DQN, Policy Gradient
- Hybrid: MCTS-MAB, MDP-VI, Hybrid combinations

**Evaluation**: Round-robin tournament with each agent playing every other agent 10 times (5 as each color)

## Limitations

We acknowledge several fairness issues favoring search algorithms over RL approaches, including unconstrained search time for tree-based methods, insufficient training episodes for RL agents, and basic neural network architectures without optimization.
