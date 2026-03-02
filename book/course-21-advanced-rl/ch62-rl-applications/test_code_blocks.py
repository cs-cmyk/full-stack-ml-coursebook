"""
Code Review Test Script for Chapter 62
Tests all code blocks in sequential order
"""

import sys
import traceback

def test_visualization_sim_to_real():
    """Test Block 1: Sim-to-Real Transfer Visualization"""
    print("\n=== Testing Block 1: Sim-to-Real Visualization ===")
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import os

        # Create diagrams directory
        os.makedirs('diagrams', exist_ok=True)

        np.random.seed(42)

        # Environment parameter (e.g., friction coefficient)
        param_range = np.linspace(0.5, 2.0, 1000)

        # Fixed simulation (narrow)
        fixed_sim = 1.0
        fixed_sim_dist = np.exp(-100 * (param_range - fixed_sim)**2)

        # Real world distribution (wider)
        real_world_mean = 1.2
        real_world_std = 0.25
        real_world_dist = np.exp(-((param_range - real_world_mean)**2) / (2 * real_world_std**2))

        # Domain randomization (broad coverage)
        dr_mean = 1.1
        dr_std = 0.4
        dr_dist = np.exp(-((param_range - dr_mean)**2) / (2 * dr_std**2))

        plt.figure(figsize=(12, 6))
        plt.plot(param_range, fixed_sim_dist / fixed_sim_dist.max(), 'b-', linewidth=2, label='Fixed Simulation')
        plt.plot(param_range, real_world_dist / real_world_dist.max(), 'r-', linewidth=2, label='Real World')
        plt.plot(param_range, dr_dist / dr_dist.max(), 'g-', linewidth=2, label='Domain Randomization')

        plt.axvline(real_world_mean, color='r', linestyle='--', alpha=0.5, label='Real World Mean')
        plt.fill_between(param_range, 0, dr_dist / dr_dist.max(), alpha=0.2, color='g')

        plt.xlabel('Environment Parameter (e.g., friction coefficient)', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title('Domain Randomization Bridges the Reality Gap', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig('diagrams/sim_to_real_coverage.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("✓ Visualization saved: diagrams/sim_to_real_coverage.png")
        return True
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        traceback.print_exc()
        return False


def test_reward_model():
    """Test Block 2: Reward Model Training"""
    print("\n=== Testing Block 2: Reward Model Training ===")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModel
        import numpy as np

        np.random.seed(42)
        torch.manual_seed(42)

        # Create a synthetic preference dataset
        class PreferenceDataset(Dataset):
            def __init__(self, n_samples=500):
                self.prompts = [
                    "Explain photosynthesis.",
                    "What is the capital of France?",
                    "How do I learn Python?",
                    "Explain quantum computing.",
                    "What causes climate change?",
                ]

                self.chosen = [
                    "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll.",
                    "The capital of France is Paris, known for the Eiffel Tower and rich cultural history.",
                    "Start with basics: variables, loops, functions. Practice with projects. Use resources like Python.org.",
                    "Quantum computing uses quantum bits (qubits) that can be in superposition, enabling parallel computation.",
                    "Climate change is primarily caused by greenhouse gas emissions from burning fossil fuels and deforestation.",
                ]

                self.rejected = [
                    "It's when plants do stuff with light.",
                    "Paris.",
                    "Just code a lot.",
                    "It's computers that use quantum stuff.",
                    "The sun and pollution.",
                ]

                self.data = []
                for _ in range(n_samples):
                    idx = np.random.randint(0, len(self.prompts))
                    self.data.append({
                        'prompt': self.prompts[idx],
                        'chosen': self.chosen[idx],
                        'rejected': self.rejected[idx]
                    })

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        # Simple reward model architecture
        class RewardModel(nn.Module):
            def __init__(self, model_name='distilbert-base-uncased'):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(model_name)
                self.reward_head = nn.Linear(self.encoder.config.hidden_size, 1)

            def forward(self, input_ids, attention_mask):
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                reward = self.reward_head(cls_embedding)
                return reward

        # Training function (shortened for testing)
        def train_reward_model(n_epochs=1, batch_size=8):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")

            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            model = RewardModel('distilbert-base-uncased').to(device)

            dataset = PreferenceDataset(n_samples=100)  # Reduced for testing
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

            model.train()
            for epoch in range(n_epochs):
                total_loss = 0
                correct = 0
                total = 0

                for batch in dataloader:
                    prompts = batch['prompt']
                    chosen = batch['chosen']
                    rejected = batch['rejected']

                    chosen_encodings = tokenizer(
                        [p + " " + c for p, c in zip(prompts, chosen)],
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors='pt'
                    ).to(device)

                    rejected_encodings = tokenizer(
                        [p + " " + r for p, r in zip(prompts, rejected)],
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors='pt'
                    ).to(device)

                    reward_chosen = model(chosen_encodings['input_ids'],
                                         chosen_encodings['attention_mask'])
                    reward_rejected = model(rejected_encodings['input_ids'],
                                           rejected_encodings['attention_mask'])

                    loss = -torch.nn.functional.logsigmoid(reward_chosen - reward_rejected).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    predictions = (reward_chosen > reward_rejected).float()
                    correct += predictions.sum().item()
                    total += len(predictions)

                avg_loss = total_loss / len(dataloader)
                accuracy = correct / total
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

            return model, tokenizer

        print("Training reward model...")
        reward_model, tokenizer = train_reward_model(n_epochs=1)

        # Test the reward model
        def test_reward_model(model, tokenizer, prompt, response_a, response_b):
            device = next(model.parameters()).device
            model.eval()

            with torch.no_grad():
                input_a = tokenizer(prompt + " " + response_a,
                                   return_tensors='pt',
                                   truncation=True,
                                   max_length=128).to(device)
                input_b = tokenizer(prompt + " " + response_b,
                                   return_tensors='pt',
                                   truncation=True,
                                   max_length=128).to(device)

                reward_a = model(input_a['input_ids'], input_a['attention_mask'])
                reward_b = model(input_b['input_ids'], input_b['attention_mask'])

                return reward_a.item(), reward_b.item()

        test_prompt = "What is machine learning?"
        response_good = "Machine learning is a field of AI where algorithms learn patterns from data to make predictions without explicit programming."
        response_bad = "It's when computers learn stuff."

        reward_good, reward_bad = test_reward_model(reward_model, tokenizer, test_prompt, response_good, response_bad)

        print(f"\nTest Results:")
        print(f"Prompt: {test_prompt}")
        print(f"Good response reward: {reward_good:.3f}")
        print(f"Bad response reward: {reward_bad:.3f}")
        print(f"Preference: {'Good response' if reward_good > reward_bad else 'Bad response'}")

        print("✓ Reward model training completed successfully")
        return True
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        traceback.print_exc()
        return False


def test_ppo_simplified():
    """Test Block 3: Simplified PPO Training"""
    print("\n=== Testing Block 3: PPO for Language Model Fine-Tuning ===")
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import numpy as np

        np.random.seed(42)
        torch.manual_seed(42)

        class SimplePPOTrainer:
            def __init__(self, policy_model, ref_model, reward_model, tokenizer,
                         kl_coef=0.1, clip_eps=0.2, lr=1e-5):
                self.policy = policy_model
                self.ref_model = ref_model
                self.reward_model = reward_model
                self.tokenizer = tokenizer
                self.kl_coef = kl_coef
                self.clip_eps = clip_eps
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

                for param in self.ref_model.parameters():
                    param.requires_grad = False
                if self.reward_model:
                    for param in self.reward_model.parameters():
                        param.requires_grad = False

            def compute_rewards(self, prompt, response):
                reward = min(len(response.split()) / 20.0, 1.0)
                quality_keywords = ['because', 'specifically', 'example', 'method', 'process']
                keyword_bonus = sum(0.1 for kw in quality_keywords if kw in response.lower())
                return reward + keyword_bonus

            def generate_response(self, prompt, max_length=50):
                inputs = self.tokenizer(prompt, return_tensors='pt')

                with torch.no_grad():
                    outputs = self.policy.generate(
                        inputs['input_ids'],
                        max_length=max_length,
                        do_sample=True,
                        top_k=50,
                        pad_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )

                response_ids = outputs.sequences[0]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                return response_text, response_ids

            def train_step(self, prompts, epochs=1):
                self.policy.train()

                all_rewards = []
                all_kl_divs = []

                for prompt in prompts:
                    response, response_ids = self.generate_response(prompt)
                    reward = self.compute_rewards(prompt, response)
                    all_rewards.append(reward)

                    inputs = self.tokenizer(prompt + response, return_tensors='pt')

                    with torch.no_grad():
                        ref_outputs = self.ref_model(**inputs)
                        ref_logits = ref_outputs.logits

                    policy_outputs = self.policy(**inputs)
                    policy_logits = policy_outputs.logits

                    policy_logprobs = F.log_softmax(policy_logits, dim=-1).mean()
                    ref_logprobs = F.log_softmax(ref_logits, dim=-1).mean()

                    kl_div = (policy_logprobs.exp() * (policy_logprobs - ref_logprobs)).abs()
                    all_kl_divs.append(kl_div.item())

                    objective = reward - self.kl_coef * kl_div
                    loss = -objective

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.optimizer.step()

                return np.mean(all_rewards), np.mean(all_kl_divs)

        print("Loading models...")
        model_name = 'distilgpt2'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        policy_model = AutoModelForCausalLM.from_pretrained(model_name)
        ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        reward_model = None

        training_prompts = [
            "Explain what deep learning is:",
            "What is reinforcement learning?",
        ]

        trainer = SimplePPOTrainer(
            policy_model=policy_model,
            ref_model=ref_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            kl_coef=0.1,
            lr=5e-6
        )

        print("\nTraining with PPO...")
        n_iterations = 1  # Reduced for testing
        for iteration in range(n_iterations):
            avg_reward, avg_kl = trainer.train_step(training_prompts)
            print(f"Iteration {iteration+1}: Avg Reward = {avg_reward:.3f}, Avg KL = {avg_kl:.4f}")

        print("✓ PPO training completed successfully")
        return True
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all code block tests"""
    print("="*70)
    print("CHAPTER 62: CODE REVIEW - Testing All Code Blocks")
    print("="*70)

    results = []

    # Test each code block
    results.append(("Sim-to-Real Visualization", test_visualization_sim_to_real()))
    results.append(("Reward Model Training", test_reward_model()))
    results.append(("PPO Training", test_ppo_simplified()))

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
