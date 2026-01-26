# Proximal Policy Optimization（PPO）算法详解  
—— 以 CartPole 环境为例

## 1. 引言

Proximal Policy Optimization（PPO）是一种经典的基于策略梯度（Policy Gradient）的强化学习算法，由 OpenAI 提出。  
PPO 的核心目标是在保证策略更新稳定性的前提下，提高训练效率，避免训练震荡和性能退化。

---

## 2. 强化学习基础

### 2.1 马尔可夫决策过程（MDP）

强化学习问题通常建模为马尔可夫决策过程：

$$\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$$

其中：

- $\mathcal{S}$：状态空间  
- $\mathcal{A}$：动作空间  
- $P(s'|s,a)$：状态转移概率  
- $R(s,a)$：奖励函数  
- $\gamma$：折扣因子  

---

### 2.2 策略与价值函数

策略定义为：

$$\pi_\theta(a|s)$$

状态价值函数：

$$V^\pi(s)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s\right]$$

动作价值函数：

$$Q^\pi(s,a)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s,a_0=a\right]$$

---

## 3. 策略梯度方法

策略优化目标为：

$$J(\theta)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right]$$

其梯度形式为：

$$\nabla_\theta J(\theta)=\mathbb{E}_t\left[\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot A_t\right]$$

其中 $A_t$ 为优势函数。

---

### 3.1 优势函数

优势函数定义为：

$$A_t=Q(s_t,a_t)-V(s_t)$$

---

## 4. PPO 算法核心

### 4.1 概率比

新旧策略的概率比定义为：

$$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}$$

---

### 4.2 PPO-Clip 目标函数

PPO 的核心目标函数为：

$$L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\left[\min\left(r_t(\theta)A_t,\ \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right]$$

---

## 5. CartPole 环境建模

### 5.1 状态空间

CartPole 状态表示为：

$$s_t=(x_t,\dot{x}_t,\theta_t,\dot{\theta}_t)$$

---

### 5.2 动作空间

动作空间为：

$$\mathcal{A}=\{0,1\}$$

---

## 6. CartPole 中的常见问题

### 6.1 小车向右边界漂移

若策略出现动作偏置，小车会持续向右运动并提前终止回合。

---

## 7. PPO 改进方法

### 7.1 熵正则化

加入策略熵以增强探索：

$$L(\theta)=L^{\mathrm{CLIP}}(\theta)+\beta\cdot\mathbb{E}_t\left[H(\pi_\theta(\cdot|s_t))\right]$$

---

### 7.2 奖励塑形（Reward Shaping）

对位置和角度进行惩罚，修正奖励函数：

$$r_t=r_{\mathrm{env}}-\alpha|x_t|-\beta|\theta_t|$$

该设计可抑制小车向边界漂移。

---

### 7.3 优势归一化

对优势函数进行标准化：

$$\hat{A}_t=\frac{A_t-\mu_A}{\sigma_A+\epsilon}$$

---

## 8. PPO 训练流程

1. 使用当前策略采样轨迹  
2. 计算回报与优势函数  
3. 固定旧策略  
4. 多轮更新 Actor 与 Critic  
5. 使用裁剪机制限制策略变化  

---

## 9. 总结

PPO 通过概率比裁剪机制，在稳定性与性能之间取得了良好平衡。  
在 CartPole 任务中，结合熵正则与奖励塑形可以有效解决边界偏移问题。

---

## 10. 参考文献

- Schulman et al., Proximal Policy Optimization Algorithms  
- Sutton and Barto, Reinforcement Learning
