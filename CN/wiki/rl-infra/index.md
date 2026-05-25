---
title: RL 基础设施
---

# RL 基础设施

RL / post-training 一侧的主题目录。

## RL 算法总览

- [[rlhf-overview]] —— RLHF：SFT + RM + PPO 三阶段流水线

  - [[rlhf-overview#三阶段流水线详解|三阶段流水线]]
  - [[rlhf-overview#数学公式推导|数学：Bradley-Terry、RM 损失、GAE]]
  - [[rlhf-overview#RLHF 的变体与演进|变体：online / offline / RLAIF / RLVR / iterative]]

- [[ppo-for-llm]] —— PPO 适配 LLM 对齐

  - [[ppo-for-llm#PPO 算法回顾|PPO 回顾（TRPO → PPO，clipped 目标）]]
  - [[ppo-for-llm#PPO 在 LLM 中的特殊适配|适配 token 级更新]]
  - [[ppo-for-llm#四模型架构|四模型架构（policy / ref / RM / value）]]
  - [[ppo-for-llm#GAE（广义优势估计）|GAE]]
  - [[ppo-for-llm#实现细节与技巧|实现技巧]]

- [[grpo]] —— GRPO：组相对策略优化（无 critic）

  - [[grpo#动机：为什么要去掉 Critic？|为什么去掉 critic]]
  - [[grpo#算法详解|算法详解]]
  - [[grpo#与 PPO 的对比|GRPO vs PPO]]
  - [[grpo#GRPO 在 DeepSeek 中的应用|DeepSeek-R1-Zero / R1 用法]]
  - [[grpo#GRPO 的变体与改进|变体（DAPO、Dr.GRPO、RLOO）]]

- [[dpo]] —— 直接偏好优化（无奖励模型）

  - [[dpo#从 RLHF 到 DPO 的推导|从 RLHF 推导]]
  - [[dpo#DPO 的变体|变体（IPO、KTO、sDPO）]]
  - [[dpo#DPO vs RLHF/PPO|DPO vs PPO]]

## 奖励建模

- [[reward-modeling]] —— 奖励模型训练与病态

  - [[reward-modeling#奖励模型架构|架构]]
  - [[reward-modeling#训练方法|训练方法]]
  - [[reward-modeling#奖励黑客 (Reward Hacking)|Reward hacking]]
  - [[reward-modeling#过程奖励模型 (PRM) vs 结果奖励模型 (ORM)|PRM vs ORM]]
  - [[reward-modeling#RLVR：可验证奖励的强化学习|RLVR（可验证奖励）]]

## RL 训练框架

- [[rl-training-frameworks]] —— 框架全景

  - [[rl-training-frameworks#OpenRLHF|OpenRLHF]]
  - [[rl-training-frameworks#TRL (Transformer Reinforcement Learning)|TRL]]
  - [[rl-training-frameworks#veRL (Volcano Engine RL)|veRL]]
  - [[rl-training-frameworks#DeepSpeed-Chat|DeepSpeed-Chat]]
  - [[rl-training-frameworks#NeMo-Aligner|NeMo-Aligner]]
  - [[rl-training-frameworks#框架对比表|框架对比表]]

## On-Policy 蒸馏（2025–2026 前沿）

- [[on-policy-distillation]] —— OPD 总伞页（起源论文、数学、变体、争论）
- [[deepseek-v4-opd]] —— DeepSeek-V4 OPD 论文精读（多教师全词表）
- [[mopd]] —— NVIDIA Nemotron-Cascade 2 MOPD 论文精读（cascade 内部 teacher）
- [[self-policy-distillation]] —— SPD 论文精读（无老师；用 correctness-aligned 梯度提取的能力子空间对 KV 激活做投影）
