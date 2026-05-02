import argparse
import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ALA_v8 import ALAV8, ALAV8Config, count_trainable_parameters


class ALACartPoleAgent(nn.Module):
    def __init__(self, max_primitives=256, device="cpu"):
        super().__init__()
        self.state_proj = nn.Linear(4, 64)  # Project 4-dim state to input_dim=64
        self.cfg = ALAV8Config(
            input_dim=64,
            output_dim=64,
            d_model=256,
            hidden_dim=512,
            max_primitives=max_primitives,
            device=device
        )
        self.ala = ALAV8(self.cfg)
        self.action_head = nn.Linear(64, 2)  # Action head: output_dim=64 -> 2 actions
        self.device = device

    def get_action(self, state: np.ndarray, context=None):
        """Sample action using current state (no x_next yet)"""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_t = self.state_proj(state_tensor)
        ala_out = self.ala(x_t, x_next=None, context=context, update_memory=False)

        y = ala_out["y"]
        action_logits = self.action_head(y)
        dist = torch.distributions.Categorical(F.softmax(action_logits, dim=-1))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, ala_out["context"]

    def process_step(self, state: np.ndarray, next_state: np.ndarray, context=None):
        """Process step with x_next for prediction loss and memory update"""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_t = self.state_proj(state_tensor)
        x_next = None
        if next_state is not None:
            next_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            x_next = self.state_proj(next_tensor)

        ala_out = self.ala(x_t, x_next=x_next, context=context, update_memory=True)
        return ala_out


def train(args):
    device = "cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = gym.make("CartPole-v1")  # Headless, no rendering
    agent = ALACartPoleAgent(max_primitives=args.max_primitives, device=device).to(device)
    total_params = count_trainable_parameters(agent)
    print(f"Total trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.lr)
    checkpoint_dir = "runtime/v8_cartpole"
    os.makedirs(checkpoint_dir, exist_ok=True)

    episode_rewards = []
    best_avg_reward = -float("inf")

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        agent.ala.reset_state()  # Reset context at start of episode
        context = None

        log_probs = []
        rewards = []
        pred_phis = []
        target_phis = []
        memory_counts = []
        settle_qualities = []
        episode_reward = 0
        done = False

        while not done:
            # Sample action (no x_next yet - note: ALA runs twice per step, see below)
            action, log_prob, context = agent.get_action(state, context)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Process step with x_next (second ALA forward pass for prediction/memory)
            next_state_np = np.array(next_state, dtype=np.float32) if not done else None
            ala_out = agent.process_step(state, next_state_np, context)

            # Collect data
            log_probs.append(log_prob)
            rewards.append(reward)
            episode_reward += reward

            if ala_out["target_phi_next"] is not None:
                pred_phis.append(ala_out["pred_phi_next"])
                target_phis.append(ala_out["target_phi_next"])

            memory_counts.append(ala_out["memory_count"].item())
            settle_qualities.append(ala_out["settle_quality"].item())
            context = ala_out["context"].detach()  # Detach context for next step
            state = next_state_np if not done else None

        episode_rewards.append(episode_reward)
        if len(episode_rewards) > 100:
            episode_rewards.pop(0)

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + args.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        # Normalize returns to reduce variance
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy loss (REINFORCE)
        log_probs = torch.stack(log_probs)
        policy_loss = -(log_probs * returns).sum()

        # Prediction loss (MSE between predicted and actual next state in phi space)
        prediction_loss = torch.tensor(0.0, device=device)
        if pred_phis:
            pred_phis = torch.stack(pred_phis)
            target_phis = torch.stack(target_phis)
            prediction_loss = F.mse_loss(pred_phis, target_phis)

        # Total loss
        total_loss = policy_loss + 0.1 * prediction_loss

        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        # Logging every 10 episodes
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            avg_memory = np.mean(memory_counts) if memory_counts else 0
            avg_settle = np.mean(settle_qualities) if settle_qualities else 0
            print(f"Episode {episode} | Avg Reward (last 10): {avg_reward:.1f} | "
                  f"Memory: {avg_memory:.0f} | Settle Quality: {avg_settle:.4f} | Loss: {total_loss.item():.4f}")

        # Check solve condition (avg >= 195 over 100 episodes)
        if len(episode_rewards) >= 100:
            avg_100 = np.mean(episode_rewards)
            if avg_100 >= 195:
                print(f"SOLVED! Average reward over 100 episodes: {avg_100:.1f}")
                torch.save({
                    "episode": episode,
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, os.path.join(checkpoint_dir, "checkpoint_solved.pt"))
                break

        # Save best checkpoint
        if episode_rewards and np.mean(episode_rewards[-10:]) > best_avg_reward:
            best_avg_reward = np.mean(episode_rewards[-10:])
            torch.save({
                "episode": episode,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_reward": best_avg_reward
            }, os.path.join(checkpoint_dir, "checkpoint_best.pt"))

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ALA v8 CartPole Agent")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-primitives", type=int, default=256, help="Max primitives for ALA")
    parser.add_argument("--cpu", action="store_true", help="Use CPU")
    args = parser.parse_args()
    train(args)
