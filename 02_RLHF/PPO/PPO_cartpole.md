# Proximal Policy Optimization（PPO）算法详解  
—— 以 CartPole 环境为例

## 1. 引言

Proximal Policy Optimization（PPO）是一种经典的 **基于策略梯度（Policy Gradient）** 的强化学习算法，由 OpenAI 提出。  
PPO 的核心目标是在 **保证策略更新稳定性** 的前提下，提高训练效率，避免传统策略梯度方法中常见的训练震荡和性能退化问题。

在本项目（AlgoLLM）中，我们使用 **CartPole-v1** 环境对 PPO 算法进行实现、分析与改进。

---

## 2. 强化学习基础回顾

### 2.1 马尔可夫决策过程（MDP）

强化学习问题通常建模为马尔可夫决策过程（MDP），由五元组组成：

$$
\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle
$$

其中：

- $\mathcal{S}$：状态空间  
- $\mathcal{A}$：动作空间  
- $P(s'|s,a)$：状态转移概率  
- $R(s,a)$：奖励函数  
- $\gamma \in (0,1)$：折扣因子  

---

### 2.2 策略与价值函数

- **策略（Policy）**：  
  $$
  \pi_\theta(a|s)
  $$

- **状态价值函数**：  
  $$
  V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r_t \mid s_0=s \right]
  $$

- **动作价值函数**：  
  $$
  Q^\pi(s,a) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r_t \mid s_0=s, a_0=a \right]
  $$

---

## 3. Policy Gradient 方法

### 3.1 策略梯度定理

策略梯度方法直接对策略参数 $\theta$ 进行优化，其目标函数为：

$$
J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

其梯度形式为：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_t
\left[
\nabla_\theta \log \pi_\theta(a_t|s_t) A_t
\right]
$$

其中：

- $A_t$ 为 Advantage，用于衡量动作相对平均水平的优劣

---

### 3.2 Advantage 函数

常见定义为：

$$
A_t = Q(s_t, a_t) - V(s_t)
$$

在实际实现中，通常使用 **GAE（Generalized Advantage Estimation）** 进行估计。

---

## 4. PPO 算法核心思想

### 4.1 为什么需要 PPO？

传统策略梯度方法存在以下问题：

- 更新步长过大，导致策略性能崩溃
- 训练过程不稳定，对学习率高度敏感

TRPO 通过 KL 约束解决该问题，但实现复杂。  
PPO 使用 **clip 技术**，在保证效果的同时大幅简化实现。

---

### 4.2 概率比（Probability Ratio）

PPO 定义新旧策略的概率比：

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

---

### 4.3 PPO-Clip 目标函数（核心公式）

$$
L^{\text{CLIP}}(\theta)
=
\mathbb{E}_t \left[
\min \left(
r_t(\theta) A_t,
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
$$

其中：

- $\epsilon$：裁剪系数（通常取 0.1 或 0.2）
- `clip` 操作限制策略更新幅度

---

## 5. CartPole 环境说明

### 5.1 状态空间

CartPole-v1 的状态由 4 个连续变量组成：

$$
s = (x,\ \dot{x},\ \theta,\ \dot{\theta})
$$

分别表示：

- 小车位置
- 小车速度
- 杆子角度
- 杆子角速度

---

### 5.2 动作空间

$$
\mathcal{A} = \{0, 1\}
$$

- 0：向左施力
- 1：向右施力

---

### 5.3 终止条件（边界问题）

环境在以下情况终止：

- 杆子倾斜角度超过阈值
- 小车位置超出左右边界

如果策略存在 **动作偏置**（例如持续向右），小车会快速撞向右边界并提前结束回合。

---

## 6. CartPole 中 PPO 的常见问题与改进

### 6.1 问题：小车持续移动到右边界

原因分析：

1. 策略初始化不平衡
2. Advantage 估计不稳定
3. 缺乏 entropy 正则项
4. 奖励结构单一

---

### 6.2 改进方法一：加入熵正则（Entropy Bonus）

在目标函数中加入熵项：

$$
L(\theta)
=
L^{\text{CLIP}}(\theta)
+
\beta \cdot \mathbb{E}_t \left[ H(\pi_\theta(\cdot|s_t)) \right]
$$

作用：

- 提高策略随机性
- 防止动作塌缩到单一方向

---

### 6.3 改进方法二：奖励塑形（Reward Shaping）

对位置和角度进行轻微惩罚：

$$
r_t
=
r_{\text{env}}
-
\alpha |x_t|
-
\beta |\theta_t|
$$

效果：

- 抑制小车向边界漂移
- 强化“居中 + 直立”的行为

---

### 6.4 改进方法三：优势归一化

$$
\hat{A}_t
=
\frac{A_t - \mu_A}{\sigma_A + \epsilon}
$$

作用：

- 降低梯度方差
- 提高训练稳定性

---

## 7. PPO 训练流程总结

完整 PPO 训练流程如下：

1. 使用当前策略采样轨迹
2. 计算回报和 Advantage
3. 固定旧策略参数
4. 多轮更新 Actor 与 Critic
5. 使用 clip 限制策略变化
6. 重复直到收敛

---

## 8. 总结

PPO 通过 **概率比裁剪机制** 在性能与稳定性之间取得了良好平衡，是当前强化学习中最常用的策略优化算法之一。

在 CartPole 这样的经典控制任务中：

- 原始 PPO 即可取得较好效果
- 加入熵正则与奖励塑形可显著改善边界偏移问题
- 为后续复杂任务（如 MuJoCo、Atari）奠定基础

---

## 9. 参考资料

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017  
- OpenAI Gym Documentation  
- Sutton & Barto, *Reinforcement Learning: An Introduction*
