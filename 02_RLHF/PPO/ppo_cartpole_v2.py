"""
PPO implementation for CartPole-v1 with GIF visualization.

核心修复点（针对“一直向右跑”的问题）：
1. 使用 Categorical + log_prob（避免错误概率更新）
2. Advantage 标准化（防止策略梯度爆炸/塌缩）
3. Entropy bonus（保证探索）
4. 合理的 clip epsilon
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import imageio

# ===============================
# 全局超参数（CartPole 安全配置）
# ===============================

ENV_NAME = "CartPole-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
LR = 3e-4

ROLLOUT_STEPS = 2048        # 每次收集多少步经验
PPO_EPOCHS = 10             # 每次 rollout 后更新多少轮
BATCH_SIZE = 64

CLIP_EPS = 0.2              # PPO clip
ENTROPY_COEF = 0.01         # 防止策略塌缩（关键）
VALUE_COEF = 0.5

TOTAL_TRAINING_STEPS = 200_000


# ===============================
# Actor-Critic 网络
# ===============================

class ActorCritic(nn.Module):
    """
    一个共享特征提取层的 Actor-Critic 网络
    - Actor 输出动作概率
    - Critic 输出状态价值 V(s)
    """

    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Actor：输出每个动作的 logits
        self.actor = nn.Linear(128, act_dim)

        # Critic：输出状态价值
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


# ===============================
# PPO Agent
# ===============================

class PPOAgent:
    def __init__(self, obs_dim, act_dim):
        self.model = ActorCritic(obs_dim, act_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def select_action(self, state):
        """
        给定状态，采样动作，并返回：
        - action
        - log_prob
        - value
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        logits, value = self.model(state)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (
            action.item(),
            log_prob.item(),
            value.item()
        )

    def evaluate(self, states, actions):
        """
        在训练阶段评估新策略：
        - log_probs
        - entropy
        - values
        """
        logits, values = self.model(states)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values.squeeze(-1)


# ===============================
# GAE / Return 计算
# ===============================

def compute_returns(rewards, dones, values, gamma):
    """
    对 CartPole（episode 较短）来说，
    使用简单 discounted return 足够稳定
    """
    returns = []
    G = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            G = 0
        G = reward + gamma * G
        returns.insert(0, G)
    return np.array(returns, dtype=np.float32)


# ===============================
# 训练主循环
# ===============================

def train():
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = PPOAgent(obs_dim, act_dim)

    state, _ = env.reset()
    step_count = 0

    # 用于收集 rollout
    states, actions, rewards, dones = [], [], [], []
    log_probs, values = [], []

    while step_count < TOTAL_TRAINING_STEPS:
        for _ in range(ROLLOUT_STEPS):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)

            state = next_state
            step_count += 1

            if done:
                state, _ = env.reset()

        print(f"Training steps: {step_count}")

        # ===============================
        # 数据整理（关键：避免 PyTorch warning）
        # ===============================

        states_t = torch.from_numpy(np.array(states)).float().to(DEVICE)
        actions_t = torch.from_numpy(np.array(actions)).long().to(DEVICE)
        old_log_probs_t = torch.from_numpy(np.array(log_probs)).float().to(DEVICE)
        values_t = torch.from_numpy(np.array(values)).float().to(DEVICE)

        returns = compute_returns(rewards, dones, values, GAMMA)
        returns_t = torch.from_numpy(returns).to(DEVICE)

        # ===============================
        # Advantage（核心修复点）
        # ===============================

        advantages = returns_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ===============================
        # PPO 更新
        # ===============================

        dataset_size = states_t.size(0)
        indices = np.arange(dataset_size)

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_idx = indices[start:end]

                batch_states = states_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns_t[batch_idx]

                new_log_probs, entropy, values_pred = agent.evaluate(
                    batch_states, batch_actions
                )

                # PPO ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = (batch_returns - values_pred).pow(2).mean()

                entropy_loss = entropy.mean()

                loss = (
                    actor_loss
                    + VALUE_COEF * critic_loss
                    - ENTROPY_COEF * entropy_loss
                )

                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

        # 清空 rollout buffer
        states.clear()
        actions.clear()
        rewards.clear()
        dones.clear()
        log_probs.clear()
        values.clear()

    env.close()
    return agent


# ===============================
# GIF 录制
# ===============================

def record_gif(agent, filename="cartpole_ppo.gif"):
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    frames = []

    state, _ = env.reset()
    done = False

    while not done:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        logits, _ = agent.model(state_t)
        action = torch.argmax(logits, dim=-1).item()

        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        frame = env.render()
        frames.append(frame)

    env.close()
    imageio.mimsave(filename, frames, fps=30)
    print(f"GIF saved to {filename}")


# ===============================
# 主入口
# ===============================

if __name__ == "__main__":
    trained_agent = train()
    record_gif(trained_agent)
