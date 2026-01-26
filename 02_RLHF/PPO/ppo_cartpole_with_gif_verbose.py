# ============================================================
# 0. 基础依赖
# ============================================================

import gymnasium as gym              # OpenAI Gym 的新版本接口
import numpy as np                   # 数值计算
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical  # 离散动作概率分布
import imageio                       # 用于生成 GIF
import os


# ============================================================
# 1. 全局超参数（决定 PPO 行为特性）
# ============================================================

ENV_NAME = "CartPole-v1"              # 经典控制问题，离散动作空间

# -------- 强化学习通用参数 --------
GAMMA = 0.99                          # 折扣因子 γ：越接近 1，越看重长期回报

# -------- GAE（优势估计）参数 --------
LAMBDA = 0.95                         # λ：控制 bias-variance tradeoff

# -------- PPO 核心参数 --------
CLIP_EPS = 0.2                        # ε：限制新旧策略更新幅度
LR = 3e-4                             # Adam 学习率
EPOCHS = 10                           # 同一批数据重复更新次数（PPO 特点）
TIMESTEPS_PER_BATCH = 2048            # 每次收集多少 on-policy 数据
MAX_TRAIN_STEPS = 200_000             # 最大训练步数

# -------- 设备选择 --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Actor–Critic 网络定义
# ============================================================
# PPO 是典型的 Actor–Critic 方法：
# - Actor：学习策略 πθ(a|s)
# - Critic：学习价值函数 V(s)
#
# 两者通常共享输入，但输出不同

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # ----------------------------------------------------
        # Actor 网络（策略网络）
        # ----------------------------------------------------
        # 输入：状态 s
        # 输出：动作概率分布 π(a|s)
        #
        # CartPole 动作为离散 {0,1}
        # 因此输出 Softmax 概率
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),  # 状态 → 隐藏层
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)         # 转为概率分布
        )

        # ----------------------------------------------------
        # Critic 网络（价值网络）
        # ----------------------------------------------------
        # 输入：状态 s
        # 输出：状态价值 V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)            # 标量输出
        )

    def act(self, state):
        """
        训练阶段使用：
        - 根据当前策略 πθ(a|s) 进行采样
        - 用于 on-policy 数据收集
        """
        probs = self.actor(state)               # πθ(a|s)
        dist = Categorical(probs)               # 构造离散分布
        action = dist.sample()                  # 随机采样动作
        log_prob = dist.log_prob(action)        # log πθ(a|s)

        return action, log_prob

    def evaluate(self, state, action):
        """
        PPO 更新阶段使用：
        - 重新计算“新策略”下的 log πθ(a|s)
        - 计算熵（entropy）
        - 计算 V(s)
        """
        probs = self.actor(state)
        dist = Categorical(probs)

        log_prob = dist.log_prob(action)        # log πθ_new(a|s)
        entropy = dist.entropy()                # 策略不确定性
        value = self.critic(state).squeeze()    # V(s)

        return log_prob, entropy, value


# ============================================================
# 3. GAE（Generalized Advantage Estimation）
# ============================================================
# 目标：计算 Advantage A_t
#
# A_t = Σ (γλ)^k δ_{t+k}
# δ_t = r_t + γ V(s_{t+1}) − V(s_t)

def compute_gae(rewards, values, dones):
    """
    输入：
        rewards: 每一步的即时奖励 r_t
        values:  Critic 给出的 V(s_t)
        dones:   episode 是否结束

    输出：
        advantages: 每一步的 A_t
    """

    advantages = []
    gae = 0.0

    # 为了统一公式，在 values 末尾补一个 0（V(s_T)=0）
    values = values + [0]

    # 从后往前递推（GAE 的关键）
    for t in reversed(range(len(rewards))):

        # TD 误差 δ_t
        delta = (
            rewards[t]
            + GAMMA * values[t + 1] * (1 - dones[t])
            - values[t]
        )

        # GAE 递推公式
        gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae

        # 插入到最前面，保持时间顺序
        advantages.insert(0, gae)

    return advantages


# ============================================================
# 4. PPO 训练主函数
# ============================================================

def train():
    """
    完整 PPO 训练流程：
    1. on-policy 采样
    2. 计算 GAE
    3. PPO clipped objective 更新策略
    """

    env = gym.make(ENV_NAME)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    state, _ = env.reset()
    total_steps = 0

    # --------------------------------------------------------
    # 主训练循环
    # --------------------------------------------------------
    while total_steps < MAX_TRAIN_STEPS:

        # --------- 轨迹缓存（on-policy）---------
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        # ====================================================
        # 1. 使用“当前策略”采样一批数据
        # ====================================================
        for _ in range(TIMESTEPS_PER_BATCH):

            state_tensor = torch.FloatTensor(state).to(DEVICE)

            # 与环境交互不需要梯度
            with torch.no_grad():
                action, log_prob = model.act(state_tensor)
                value = model.critic(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # 保存轨迹
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value.item())

            state = next_state
            total_steps += 1

            if done:
                state, _ = env.reset()

        # ====================================================
        # 2. 计算 Advantage 与 Return
        # ====================================================
        advantages = compute_gae(rewards, values, dones)
        returns = np.array(advantages) + np.array(values)

        # 转为 Tensor
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        old_log_probs = torch.FloatTensor(log_probs).to(DEVICE)
        advantages = torch.FloatTensor(advantages).to(DEVICE)
        returns = torch.FloatTensor(returns).to(DEVICE)

        # Advantage 标准化（极其重要）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ====================================================
        # 3. PPO 多 epoch 更新
        # ====================================================
        for _ in range(EPOCHS):

            # 新策略下的 log πθ(a|s)、V(s)
            log_probs, entropy, values = model.evaluate(states, actions)

            # πθ(a|s) / πθ_old(a|s)
            ratios = torch.exp(log_probs - old_log_probs)

            # PPO clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(
                ratios,
                1 - CLIP_EPS,
                1 + CLIP_EPS
            ) * advantages

            # Actor：最大化 surrogate → 最小化负值
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic：回归真实 return
            critic_loss = nn.MSELoss()(values, returns)

            # Entropy bonus：防止过早收敛
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Training steps: {total_steps}")

    env.close()
    return model


# ============================================================
# 5. GIF 可视化函数（评估阶段）
# ============================================================

def record_gif(model, gif_path="cartpole_ppo.gif", max_steps=500):
    """
    使用训练好的策略生成 CartPole 运行 GIF
    """

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    frames = []

    state, _ = env.reset()

    for _ in range(max_steps):
        # 获取当前画面帧
        frames.append(env.render())

        state_tensor = torch.FloatTensor(state).to(DEVICE)

        # 评估阶段使用贪心策略（更稳定）
        with torch.no_grad():
            probs = model.actor(state_tensor)
            action = torch.argmax(probs).item()

        state, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    env.close()

    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIF saved at: {os.path.abspath(gif_path)}")


# ============================================================
# 6. 程序入口
# ============================================================

if __name__ == "__main__":
    model = train()
    record_gif(model)
