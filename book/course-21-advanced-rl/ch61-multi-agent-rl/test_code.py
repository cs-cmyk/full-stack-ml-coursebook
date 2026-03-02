#!/usr/bin/env python3
"""
Test script for code blocks in ch61/content.md
"""

import sys
import traceback

def test_block_1_qmix():
    """Test QMIX implementation"""
    print("\n" + "="*60)
    print("Testing Block 1: QMIX Implementation")
    print("="*60)

    try:
        # Cooperative Multi-Agent RL: QMIX on a simple gridworld
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from collections import deque
        import random
        import matplotlib.pyplot as plt

        # Set random seeds
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)

        # Simple 5x5 gridworld where 2 agents must collect items cooperatively
        class CooperativeGridWorld:
            def __init__(self, size=5, n_agents=2, n_items=3):
                self.size = size
                self.n_agents = n_agents
                self.n_items = n_items
                self.reset()

            def reset(self):
                # Random agent positions
                self.agent_positions = []
                for _ in range(self.n_agents):
                    pos = (np.random.randint(0, self.size),
                           np.random.randint(0, self.size))
                    self.agent_positions.append(pos)

                # Random item positions
                self.item_positions = []
                for _ in range(self.n_items):
                    pos = (np.random.randint(0, self.size),
                           np.random.randint(0, self.size))
                    self.item_positions.append(pos)

                self.items_collected = 0
                self.steps = 0
                return self.get_obs()

            def get_obs(self):
                # Each agent observes: own position + all item positions (normalized)
                obs = []
                for agent_pos in self.agent_positions:
                    agent_obs = [agent_pos[0]/self.size, agent_pos[1]/self.size]
                    for item_pos in self.item_positions:
                        agent_obs.extend([item_pos[0]/self.size, item_pos[1]/self.size])
                    obs.append(np.array(agent_obs, dtype=np.float32))
                return obs

            def get_state(self):
                # Global state: all agent positions + all item positions
                state = []
                for pos in self.agent_positions:
                    state.extend([pos[0]/self.size, pos[1]/self.size])
                for pos in self.item_positions:
                    state.extend([pos[0]/self.size, pos[1]/self.size])
                return np.array(state, dtype=np.float32)

            def step(self, actions):
                # Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
                self.steps += 1
                moves = [(0,-1), (0,1), (-1,0), (1,0), (0,0)]

                # Move agents
                for i, action in enumerate(actions):
                    move = moves[action]
                    new_pos = (self.agent_positions[i][0] + move[0],
                               self.agent_positions[i][1] + move[1])
                    # Clip to grid boundaries
                    new_pos = (max(0, min(self.size-1, new_pos[0])),
                               max(0, min(self.size-1, new_pos[1])))
                    self.agent_positions[i] = new_pos

                # Check item collection (any agent on item position)
                reward = 0
                items_to_remove = []
                for item_idx, item_pos in enumerate(self.item_positions):
                    for agent_pos in self.agent_positions:
                        if agent_pos == item_pos:
                            reward += 1
                            items_to_remove.append(item_idx)
                            self.items_collected += 1
                            break

                # Remove collected items
                for idx in sorted(items_to_remove, reverse=True):
                    del self.item_positions[idx]

                # Bonus reward if agents are close (encourage coordination)
                if len(self.agent_positions) == 2:
                    dist = abs(self.agent_positions[0][0] - self.agent_positions[1][0]) + \
                           abs(self.agent_positions[0][1] - self.agent_positions[1][1])
                    if dist <= 2:
                        reward += 0.1

                done = len(self.item_positions) == 0 or self.steps >= 50
                return self.get_obs(), reward, done

        # Agent Q-network
        class AgentQNetwork(nn.Module):
            def __init__(self, obs_dim, n_actions):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(obs_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, n_actions)
                )

            def forward(self, obs):
                return self.fc(obs)

        # QMIX mixing network
        class QMixNetwork(nn.Module):
            def __init__(self, n_agents, state_dim):
                super().__init__()
                self.n_agents = n_agents

                # Hypernetwork for generating mixing weights
                self.hyper_w1 = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, n_agents * 32)
                )

                self.hyper_w2 = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32)
                )

                self.hyper_b1 = nn.Linear(state_dim, 32)
                self.hyper_b2 = nn.Sequential(
                    nn.Linear(state_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )

            def forward(self, agent_qs, state):
                # agent_qs: (batch_size, n_agents)
                # state: (batch_size, state_dim)
                batch_size = agent_qs.size(0)

                # Generate weights (ensure non-negative via abs)
                w1 = torch.abs(self.hyper_w1(state))
                w1 = w1.view(batch_size, self.n_agents, 32)

                b1 = self.hyper_b1(state)
                b1 = b1.view(batch_size, 1, 32)

                # First layer: mix agent Q-values
                hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
                hidden = torch.relu(hidden)

                # Second layer
                w2 = torch.abs(self.hyper_w2(state))
                w2 = w2.view(batch_size, 32, 1)

                b2 = self.hyper_b2(state)
                b2 = b2.view(batch_size, 1, 1)

                q_tot = torch.bmm(hidden, w2) + b2
                return q_tot.squeeze()

        # QMIX Agent
        class QMIXAgent:
            def __init__(self, env, lr=1e-3, gamma=0.99, epsilon_start=1.0,
                         epsilon_end=0.05, epsilon_decay=0.995):
                self.env = env
                self.n_agents = env.n_agents
                self.n_actions = 5
                self.gamma = gamma
                self.epsilon = epsilon_start
                self.epsilon_end = epsilon_end
                self.epsilon_decay = epsilon_decay

                # Observation and state dimensions
                obs = env.reset()
                self.obs_dim = len(obs[0])
                self.state_dim = len(env.get_state())

                # Agent Q-networks
                self.agent_networks = [AgentQNetwork(self.obs_dim, self.n_actions)
                                       for _ in range(self.n_agents)]

                # QMIX mixing network
                self.mixer = QMixNetwork(self.n_agents, self.state_dim)

                # Target networks
                self.target_agent_networks = [AgentQNetwork(self.obs_dim, self.n_actions)
                                              for _ in range(self.n_agents)]
                self.target_mixer = QMixNetwork(self.n_agents, self.state_dim)
                self.update_target_networks()

                # Optimizer
                params = []
                for net in self.agent_networks:
                    params += list(net.parameters())
                params += list(self.mixer.parameters())
                self.optimizer = optim.Adam(params, lr=lr)

                # Replay buffer
                self.buffer = deque(maxlen=10000)

            def update_target_networks(self):
                for target_net, net in zip(self.target_agent_networks, self.agent_networks):
                    target_net.load_state_dict(net.state_dict())
                self.target_mixer.load_state_dict(self.mixer.state_dict())

            def select_actions(self, obs, explore=True):
                actions = []
                for i, o in enumerate(obs):
                    if explore and np.random.rand() < self.epsilon:
                        actions.append(np.random.randint(0, self.n_actions))
                    else:
                        with torch.no_grad():
                            o_tensor = torch.FloatTensor(o).unsqueeze(0)
                            q_values = self.agent_networks[i](o_tensor)
                            actions.append(q_values.argmax().item())
                return actions

            def store_transition(self, obs, actions, reward, next_obs, done, state, next_state):
                self.buffer.append((obs, actions, reward, next_obs, done, state, next_state))

            def train(self, batch_size=32):
                if len(self.buffer) < batch_size:
                    return 0

                batch = random.sample(self.buffer, batch_size)

                # Prepare batch tensors
                obs_batch = [[] for _ in range(self.n_agents)]
                next_obs_batch = [[] for _ in range(self.n_agents)]
                actions_batch = [[] for _ in range(self.n_agents)]
                rewards_batch = []
                dones_batch = []
                states_batch = []
                next_states_batch = []

                for transition in batch:
                    obs, actions, reward, next_obs, done, state, next_state = transition
                    for i in range(self.n_agents):
                        obs_batch[i].append(obs[i])
                        next_obs_batch[i].append(next_obs[i])
                        actions_batch[i].append(actions[i])
                    rewards_batch.append(reward)
                    dones_batch.append(done)
                    states_batch.append(state)
                    next_states_batch.append(next_state)

                # Convert to tensors
                obs_tensors = [torch.FloatTensor(np.array(obs_batch[i])) for i in range(self.n_agents)]
                next_obs_tensors = [torch.FloatTensor(np.array(next_obs_batch[i]))
                                   for i in range(self.n_agents)]
                actions_tensors = [torch.LongTensor(actions_batch[i]) for i in range(self.n_agents)]
                rewards_tensor = torch.FloatTensor(rewards_batch)
                dones_tensor = torch.FloatTensor(dones_batch)
                states_tensor = torch.FloatTensor(np.array(states_batch))
                next_states_tensor = torch.FloatTensor(np.array(next_states_batch))

                # Current Q-values
                agent_qs = []
                for i in range(self.n_agents):
                    q_vals = self.agent_networks[i](obs_tensors[i])
                    q_vals = q_vals.gather(1, actions_tensors[i].unsqueeze(1)).squeeze()
                    agent_qs.append(q_vals)
                agent_qs = torch.stack(agent_qs, dim=1)

                # Current Q_tot
                q_tot = self.mixer(agent_qs, states_tensor)

                # Target Q-values
                with torch.no_grad():
                    target_agent_qs = []
                    for i in range(self.n_agents):
                        target_q_vals = self.target_agent_networks[i](next_obs_tensors[i])
                        target_q_vals = target_q_vals.max(dim=1)[0]
                        target_agent_qs.append(target_q_vals)
                    target_agent_qs = torch.stack(target_agent_qs, dim=1)

                    target_q_tot = self.target_mixer(target_agent_qs, next_states_tensor)
                    targets = rewards_tensor + self.gamma * target_q_tot * (1 - dones_tensor)

                # Loss and optimization
                loss = nn.MSELoss()(q_tot, targets)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 10)
                for net in self.agent_networks:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
                self.optimizer.step()

                return loss.item()

        # Training loop (reduced episodes for testing)
        env = CooperativeGridWorld(size=5, n_agents=2, n_items=3)
        agent = QMIXAgent(env)

        n_episodes = 50  # Reduced from 500 for testing
        episode_rewards = []
        losses = []

        print("Training QMIX agent...")
        for episode in range(n_episodes):
            obs = env.reset()
            state = env.get_state()
            episode_reward = 0
            done = False

            while not done:
                actions = agent.select_actions(obs, explore=True)
                next_obs, reward, done = env.step(actions)
                next_state = env.get_state()

                agent.store_transition(obs, actions, reward, next_obs, done, state, next_state)

                loss = agent.train()
                if loss > 0:
                    losses.append(loss)

                obs = next_obs
                state = next_state
                episode_reward += reward

            episode_rewards.append(episode_reward)
            agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)

            # Update target networks
            if episode % 10 == 0:
                agent.update_target_networks()

            if episode % 25 == 0:
                avg_reward = np.mean(episode_rewards[-25:]) if len(episode_rewards) >= 25 else np.mean(episode_rewards)
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Rewards
        axes[0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
        # Moving average
        window = 10
        if len(episode_rewards) >= window:
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(episode_rewards)), moving_avg,
                        label=f'{window}-Episode Moving Avg', linewidth=2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('QMIX Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss
        if losses:
            axes[1].plot(losses, alpha=0.5)
            axes[1].set_xlabel('Training Step')
            axes[1].set_ylabel('TD Loss')
            axes[1].set_title('Training Loss')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('book/course-21/ch61/qmix_training.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved training plot to book/course-21/ch61/qmix_training.png")
        print("✓ Block 1 PASSED")
        return True

    except Exception as e:
        print(f"✗ Block 1 FAILED: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []

    # Test Block 1
    results.append(("Block 1: QMIX", test_block_1_qmix()))

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} blocks passed")

    sys.exit(0 if all(p for _, p in results) else 1)
