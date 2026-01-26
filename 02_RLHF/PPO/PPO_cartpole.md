# Proximal Policy Optimization（PPO）算法说明  
—— 以 CartPole 环境为例的实现与改进

## 1. PPO 算法概述

Proximal Policy Optimization（PPO）是一种 **基于策略梯度（Policy Gradient）** 的强化学习算法，由 OpenAI 提出，核心目标是在 **保证策略更新稳定性** 的前提下，提高样本利用效率。

PPO 广泛应用于：
- 连续 / 离散动作空间
- 大规模强化学习
- RLHF（Reinforcement Learning from Human Feedback）

---

## 2. 策略梯度的基本形式

在策略梯度方法中，我们希望最大化期望回报：

\[
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} r_t \right]
\]

其梯度形式为：

\[
\nabla_\theta J(\theta)
= \mathbb{E}_t \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A_t \right]
\]

其中：
- \( \pi_\theta(a_t | s_t) \)：策略网络
- \( A_t \)：优势函数（Advantage）

---

## 3. PPO 的核心思想：限制策略更新幅度

### 3.1 概率比率（Probability Ratio）

PPO 使用 **新旧策略的概率比率** 来衡量更新幅度：

\[
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
\]

---

### 3.2 PPO Clip Objective（核心公式）

PPO 的目标函数定义为：

\[
L^{\text{CLIP}}(\theta)
= \mathbb{E}_t \left[
\min \left(
r_t(\theta) A_t,\;
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
\]

其中：
- \( \epsilon \) 为 clip 参数（通常为 0.1–0.2）
- 防止策略在一次更新中发生过大偏移

---

## 4. Actor-Critic 架构

PPO 通常采用 **Actor-Critic 结构**：

- **Actor（策略网络）**  
  输出动作分布 \( \pi(a|s) \)

- **Critic（价值网络）**  
  估计状态价值函数：

\[
V(s_t) = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \right]
\]

---

## 5. Advantage 的计算与标准化（关键）

### 5.1 Advantage 定义

在 CartPole 实现中使用：

\[
A_t = R_t - V(s_t)
\]

其中 \( R_t \) 为折扣回报（return）。

---

### 5.2 为什么必须进行 Advantage 标准化？

如果不做标准化：
- PPO 的 clip 机制会失效
- 策略梯度会向某一动作极度偏移
- 容易出现 **策略塌缩（Policy Collapse）**

标准化公式：

\[
\hat{A}_t = \frac{A_t - \mu(A)}{\sigma(A) + \epsilon}
\]

这是解决 CartPole **“只向一个方向跑”问题的核心修复点之一**。

---

## 6. Entropy Bonus：防止策略塌缩

### 6.1 Entropy 的定义

对离散动作空间，策略熵为：

\[
H(\pi(\cdot|s)) = - \sum_a \pi(a|s) \log \pi(a|s)
\]

---

### 6.2 PPO 的完整损失函数

\[
L = 
L^{\text{CLIP}}
+ c_v \cdot L^{\text{value}}
- c_e \cdot H(\pi)
\]

其中：
- \( L^{\text{value}} = (R_t - V(s_t))^2 \)
- \( c_e \) 为 entropy 系数（CartPole 中推荐 0.01）

**Entropy Bonus 是防止策略过早确定为单一动作的关键机制。**

---

## 7. CartPole 环境简介

- 状态空间（4 维）：
  \[
  (x, \dot{x}, \theta, \dot{\theta})
  \]

- 动作空间（离散）：
  - 0：向左施力
  - 1：向右施力

- 奖励：
  - 每存活一步 +1

---

## 8. 初版 CartPole PPO 的常见问题

在初版实现中，常见问题包括：

1. **没有 Advantage 标准化**
2. **Entropy 系数为 0**
3. **log\_prob 计算方式错误**
4. **策略更新过大（clip 过宽）**

### 典型现象

- CartPole 小车 **始终向右（或向左）移动**
- 很快撞到边界
- 动作概率塌缩为：
  \[
  \pi(a=1|s) \approx 1
  \]

---

## 9. 改进版 CartPole PPO 的关键修复

### 9.1 正确的动作采样方式

```python
dist = Categorical(logits=logits)
action = dist.sample()
log_prob = dist.log_prob(action)
