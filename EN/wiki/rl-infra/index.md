---
title: RL Infrastructure
---

# RL Infrastructure

Topical index for the RL / post-training side.

## RL algorithm overviews

- [[rlhf-overview]] — RLHF: three-stage SFT + RM + PPO pipeline

  - [[rlhf-overview#The Three-Stage Pipeline|The three-stage pipeline]]
  - [[rlhf-overview#Mathematical Derivations|Math: Bradley-Terry, RM loss, GAE]]
  - [[rlhf-overview#RLHF Variants and Evolution|Variants: online/offline/RLAIF/RLVR/iterative]]

- [[ppo-for-llm]] — PPO adapted to LLM alignment

  - [[ppo-for-llm#PPO Algorithm Recap|PPO recap (TRPO → PPO, clipped objective)]]
  - [[ppo-for-llm#Adapting PPO to LLMs|Adapting PPO to token-level updates]]
  - [[ppo-for-llm#Four-Model Architecture|Four-model architecture (policy / ref / RM / value)]]
  - [[ppo-for-llm#GAE (Generalized Advantage Estimation)|GAE]]
  - [[ppo-for-llm#Implementation Details and Tricks|Implementation tricks]]

- [[grpo]] — GRPO: group-relative policy optimization (no critic)

  - [[grpo#Motivation: Why Remove the Critic?|Why remove the critic]]
  - [[grpo#Algorithm in Detail|Algorithm details]]
  - [[grpo#Comparison with PPO|GRPO vs PPO]]
  - [[grpo#GRPO Applied at DeepSeek|DeepSeek-R1-Zero / R1 usage]]
  - [[grpo#GRPO Variants and Improvements|Variants (DAPO, Dr.GRPO, RLOO)]]

- [[dpo]] — Direct Preference Optimization (no reward model)

  - [[dpo#Derivation: From RLHF to DPO|Derivation from RLHF]]
  - [[dpo#DPO Variants|Variants (IPO, KTO, sDPO)]]
  - [[dpo#DPO vs RLHF/PPO|DPO vs PPO]]

## Reward modeling

- [[reward-modeling]] — reward model training and pathologies

  - [[reward-modeling#Reward Model Architecture|Architectures]]
  - [[reward-modeling#Training Methods|Training methods]]
  - [[reward-modeling#Reward Hacking|Reward hacking]]
  - [[reward-modeling#Process Reward Models (PRM) vs. Outcome Reward Models (ORM)|PRM vs ORM]]
  - [[reward-modeling#RLVR: Reinforcement Learning from Verifiable Rewards|RLVR (verifiable rewards)]]

## RL training frameworks

- [[rl-training-frameworks]] — frameworks landscape

  - [[rl-training-frameworks#OpenRLHF|OpenRLHF]]
  - [[rl-training-frameworks#TRL (Transformer Reinforcement Learning)|TRL]]
  - [[rl-training-frameworks#veRL (Volcano Engine RL)|veRL]]
  - [[rl-training-frameworks#DeepSpeed-Chat|DeepSpeed-Chat]]
  - [[rl-training-frameworks#NeMo-Aligner|NeMo-Aligner]]
  - [[rl-training-frameworks#Framework Comparison|Framework comparison table]]

## On-policy distillation (2025–2026 frontier)

- [[on-policy-distillation]] — OPD umbrella page (origin paper, math, variants, debate)
- [[deepseek-v4-opd]] — DeepSeek-V4 OPD paper review (multi-teacher full-vocabulary)
- [[mopd]] — NVIDIA Nemotron-Cascade 2 MOPD paper review (cascade-internal teachers)
- [[self-policy-distillation]] — SPD paper review (teacher-free; KV-activation subspace projection from correctness-aligned gradients)
