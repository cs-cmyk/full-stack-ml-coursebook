"""
Test remaining code blocks from Chapter 62
"""

import sys
import traceback
import warnings
warnings.filterwarnings('ignore')


def test_dpo():
    """Test Block 4: Direct Preference Optimization"""
    print("\n=== Testing Block 4: DPO ===")
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import numpy as np

        np.random.seed(42)
        torch.manual_seed(42)

        class DPOPreferenceDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def compute_dpo_loss(policy_model, ref_model, tokenizer, batch, beta=0.1):
            device = next(policy_model.parameters()).device

            prompts = batch['prompt']
            chosen = batch['chosen']
            rejected = batch['rejected']

            chosen_inputs = tokenizer(
                [p + c for p, c in zip(prompts, chosen)],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)

            rejected_inputs = tokenizer(
                [p + r for p, r in zip(prompts, rejected)],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)

            policy_chosen_outputs = policy_model(**chosen_inputs)
            policy_rejected_outputs = policy_model(**rejected_inputs)

            with torch.no_grad():
                ref_chosen_outputs = ref_model(**chosen_inputs)
                ref_rejected_outputs = ref_model(**rejected_inputs)

            policy_chosen_logprobs = F.log_softmax(policy_chosen_outputs.logits, dim=-1).mean(dim=(1, 2))
            policy_rejected_logprobs = F.log_softmax(policy_rejected_outputs.logits, dim=-1).mean(dim=(1, 2))

            ref_chosen_logprobs = F.log_softmax(ref_chosen_outputs.logits, dim=-1).mean(dim=(1, 2))
            ref_rejected_logprobs = F.log_softmax(ref_rejected_outputs.logits, dim=-1).mean(dim=(1, 2))

            chosen_log_ratio = policy_chosen_logprobs - ref_chosen_logprobs
            rejected_log_ratio = policy_rejected_logprobs - ref_rejected_logprobs

            logits = beta * (chosen_log_ratio - rejected_log_ratio)
            loss = -F.logsigmoid(logits).mean()

            implicit_reward_chosen = beta * chosen_log_ratio
            implicit_reward_rejected = beta * rejected_log_ratio

            return loss, implicit_reward_chosen.mean().item(), implicit_reward_rejected.mean().item()

        # Prepare synthetic preference dataset (shortened)
        preference_data = [
            {
                'prompt': 'What is machine learning?',
                'chosen': ' Machine learning is a field of AI where algorithms learn patterns from data to make predictions.',
                'rejected': ' It is when computers learn things.'
            },
            {
                'prompt': 'Explain neural networks.',
                'chosen': ' Neural networks are computational models inspired by biological neurons that process information in layers.',
                'rejected': ' Networks with neurons.'
            },
        ] * 10

        print("Loading models for DPO training...")
        model_name = 'distilgpt2'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        policy_model = AutoModelForCausalLM.from_pretrained(model_name)
        ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        for param in ref_model.parameters():
            param.requires_grad = False

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy_model = policy_model.to(device)
        ref_model = ref_model.to(device)

        dataset = DPOPreferenceDataset(preference_data)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)
        beta = 0.1

        print(f"\nTraining with DPO (beta={beta})...")
        n_epochs = 1  # Reduced for testing
        for epoch in range(n_epochs):
            total_loss = 0
            total_reward_chosen = 0
            total_reward_rejected = 0
            n_batches = 0

            policy_model.train()
            for batch in dataloader:
                loss, reward_chosen, reward_rejected = compute_dpo_loss(
                    policy_model, ref_model, tokenizer, batch, beta
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_reward_chosen += reward_chosen
                total_reward_rejected += reward_rejected
                n_batches += 1

            avg_loss = total_loss / n_batches
            avg_reward_chosen = total_reward_chosen / n_batches
            avg_reward_rejected = total_reward_rejected / n_batches

            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Implicit Reward (chosen): {avg_reward_chosen:.4f}")
            print(f"  Implicit Reward (rejected): {avg_reward_rejected:.4f}")
            print(f"  Reward Gap: {avg_reward_chosen - avg_reward_rejected:.4f}")

        print("✓ DPO training completed successfully")
        return True
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        traceback.print_exc()
        return False


def test_sim_to_real():
    """Test Block 5: Sim-to-Real with Domain Randomization"""
    print("\n=== Testing Block 5: Sim-to-Real Transfer ===")
    try:
        import numpy as np
        import gym
        from gym import spaces
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        np.random.seed(42)

        class SimpleReachingEnv(gym.Env):
            def __init__(self, randomize_dynamics=False):
                super().__init__()
                self.randomize_dynamics = randomize_dynamics
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
                self.observation_space = spaces.Box(
                    low=-10.0, high=10.0, shape=(6,), dtype=np.float32
                )
                self.reset_physics()
                self.target = np.array([3.0, 3.0])
                self.max_steps = 50
                self.current_step = 0

            def reset_physics(self):
                if self.randomize_dynamics:
                    self.mass = np.random.uniform(0.8, 1.2)
                    self.friction = np.random.uniform(0.05, 0.15)
                else:
                    self.mass = 1.0
                    self.friction = 0.1

            def reset(self):
                self.reset_physics()
                self.position = np.random.uniform(-1.0, 1.0, size=2)
                self.velocity = np.zeros(2)
                self.current_step = 0
                return self._get_obs()

            def _get_obs(self):
                return np.concatenate([
                    self.position,
                    self.velocity,
                    self.target
                ]).astype(np.float32)

            def step(self, action):
                acceleration = action / self.mass
                self.velocity = self.velocity * (1.0 - self.friction) + acceleration * 0.1
                self.position = self.position + self.velocity * 0.1

                distance = np.linalg.norm(self.position - self.target)
                reward = -distance

                if distance < 0.5:
                    reward += 10.0
                    done = True
                else:
                    done = False

                self.current_step += 1
                if self.current_step >= self.max_steps:
                    done = True

                info = {'distance': distance, 'mass': self.mass, 'friction': self.friction}

                return self._get_obs(), reward, done, info

        print("Training agent WITHOUT domain randomization...")
        env_fixed = DummyVecEnv([lambda: SimpleReachingEnv(randomize_dynamics=False)])
        agent_fixed = PPO('MlpPolicy', env_fixed, verbose=0, seed=42)
        agent_fixed.learn(total_timesteps=5000)  # Reduced for testing
        print("Training complete (fixed physics).")

        print("\nTraining agent WITH domain randomization...")
        env_randomized = DummyVecEnv([lambda: SimpleReachingEnv(randomize_dynamics=True)])
        agent_randomized = PPO('MlpPolicy', env_randomized, verbose=0, seed=42)
        agent_randomized.learn(total_timesteps=5000)  # Reduced for testing
        print("Training complete (randomized physics).")

        # Quick evaluation
        def evaluate_agent(agent, n_episodes=5):
            test_env = SimpleReachingEnv(randomize_dynamics=True)
            successes = 0
            for _ in range(n_episodes):
                obs = test_env.reset()
                done = False
                while not done:
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, done, info = test_env.step(action)
                if info['distance'] < 0.5:
                    successes += 1
            return successes / n_episodes

        success_fixed = evaluate_agent(agent_fixed, n_episodes=5)
        success_randomized = evaluate_agent(agent_randomized, n_episodes=5)

        print(f"\nResults:")
        print(f"Fixed Physics Training -> Success Rate: {success_fixed:.2%}")
        print(f"Domain Randomization Training -> Success Rate: {success_randomized:.2%}")

        print("✓ Sim-to-real training completed successfully")
        return True
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        traceback.print_exc()
        return False


def test_tsp():
    """Test Block 6: TSP with RL"""
    print("\n=== Testing Block 6: RL for TSP ===")
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        np.random.seed(42)
        torch.manual_seed(42)

        class AttentionTSPSolver(nn.Module):
            def __init__(self, embedding_dim=128):
                super().__init__()
                self.embedding_dim = embedding_dim
                self.embedding = nn.Linear(2, embedding_dim)
                self.query = nn.Linear(embedding_dim, embedding_dim)
                self.key = nn.Linear(embedding_dim, embedding_dim)
                self.decoder = nn.GRU(embedding_dim, embedding_dim, batch_first=True)

            def forward(self, city_coords, mask=None):
                batch_size, n_cities, _ = city_coords.shape
                embedded = torch.tanh(self.embedding(city_coords))
                h = embedded[:, 0:1, :]

                if mask is None:
                    mask = torch.ones(batch_size, n_cities, device=city_coords.device)
                    mask[:, 0] = 0

                tour = [0]
                log_probs = []

                for step in range(n_cities - 1):
                    q = self.query(h)
                    k = self.key(embedded)
                    logits = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(self.embedding_dim)
                    logits = logits.squeeze(1)
                    logits = logits.masked_fill(mask == 0, float('-inf'))
                    probs = F.softmax(logits, dim=-1)

                    if self.training:
                        dist = torch.distributions.Categorical(probs)
                        next_city = dist.sample()
                        log_prob = dist.log_prob(next_city)
                    else:
                        next_city = torch.argmax(probs, dim=-1)
                        log_prob = torch.log(probs.gather(1, next_city.unsqueeze(1)).squeeze(1) + 1e-10)

                    tour.append(next_city.item())
                    mask[0, next_city] = 0
                    log_probs.append(log_prob)

                    selected_embedding = embedded[:, next_city, :].unsqueeze(1)
                    _, h = self.decoder(selected_embedding, h.transpose(0, 1))
                    h = h.transpose(0, 1)

                tour.append(0)
                return tour, torch.stack(log_probs)

        def compute_tour_length(city_coords, tour):
            length = 0
            for i in range(len(tour) - 1):
                city_a = city_coords[tour[i]]
                city_b = city_coords[tour[i + 1]]
                length += np.linalg.norm(city_a - city_b)
            return length

        print("Training attention-based TSP solver...")
        model = AttentionTSPSolver(embedding_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        n_cities = 10
        n_epochs = 5  # Reduced for testing

        for epoch in range(n_epochs):
            epoch_loss = 0
            epoch_length = 0

            for _ in range(10):  # Mini-batches
                city_coords = np.random.rand(n_cities, 2)
                city_coords_tensor = torch.FloatTensor(city_coords).unsqueeze(0)

                model.train()
                tour, log_probs = model(city_coords_tensor)

                tour_length = compute_tour_length(city_coords, tour)
                reward = -tour_length

                baseline = -3.0
                loss = -(log_probs.sum() * (reward - baseline))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_length += tour_length

            avg_loss = epoch_loss / 10
            avg_length = epoch_length / 10

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.3f}, Avg Tour Length: {avg_length:.3f}")

        print("✓ TSP training completed successfully")
        return True
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        traceback.print_exc()
        return False


def test_solution_1():
    """Test Solution 1: Reward Model Analysis"""
    print("\n=== Testing Solution 1: Reward Model with Sentiment ===")
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModel

        np.random.seed(42)
        torch.manual_seed(42)

        class SyntheticPreferenceDataset(Dataset):
            def __init__(self, n_samples=50):
                self.data = []
                prompts = ["Explain the concept:", "Describe the process:", "What is your opinion on:"]

                for _ in range(n_samples):
                    prompt = np.random.choice(prompts) + " [topic]"
                    chosen = self._generate_response(length='long', sentiment='positive')
                    rejected = self._generate_response(length='short', sentiment='negative')
                    self.data.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected})

            def _generate_response(self, length, sentiment):
                if length == 'long':
                    base = "This is an excellent and comprehensive explanation that provides valuable insights. "
                    response = base * 3
                else:
                    response = "Short answer here."

                if sentiment == 'positive':
                    response += " This is wonderful and highly beneficial."
                elif sentiment == 'negative':
                    response += " This is problematic and concerning."

                return response

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        class AnalyzableRewardModel(nn.Module):
            def __init__(self, model_name='distilbert-base-uncased'):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(model_name)
                self.reward_head = nn.Linear(self.encoder.config.hidden_size, 1)

            def forward(self, input_ids, attention_mask, return_attention=False):
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                      output_attentions=return_attention)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                reward = self.reward_head(cls_embedding)

                if return_attention:
                    return reward, outputs.attentions
                return reward

        dataset = SyntheticPreferenceDataset(n_samples=50)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        model = AnalyzableRewardModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        print("Training reward model on length + sentiment preferences...")
        model.train()
        for epoch in range(1):
            total_loss = 0
            for batch in dataloader:
                prompts, chosen, rejected = batch['prompt'], batch['chosen'], batch['rejected']

                chosen_inputs = tokenizer([p + " " + c for p, c in zip(prompts, chosen)],
                                          padding=True, truncation=True, max_length=128,
                                          return_tensors='pt')
                rejected_inputs = tokenizer([p + " " + r for p, r in zip(prompts, rejected)],
                                            padding=True, truncation=True, max_length=128,
                                            return_tensors='pt')

                reward_chosen = model(chosen_inputs['input_ids'], chosen_inputs['attention_mask'])
                reward_rejected = model(rejected_inputs['input_ids'], rejected_inputs['attention_mask'])

                loss = -torch.nn.functional.logsigmoid(reward_chosen - reward_rejected).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

        print("✓ Solution 1 completed successfully")
        return True
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all remaining tests"""
    print("="*70)
    print("CHAPTER 62: TESTING REMAINING CODE BLOCKS")
    print("="*70)

    results = []

    results.append(("DPO", test_dpo()))
    results.append(("Sim-to-Real", test_sim_to_real()))
    results.append(("TSP with RL", test_tsp()))
    results.append(("Solution 1", test_solution_1()))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
