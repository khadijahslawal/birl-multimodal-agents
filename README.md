# Inferring Information-Gathering Preferences in Multimodal Agents via Bayesian Inverse Reinforcement Learning

**TTIC 31170 — Planning, Learning, and Estimation for Robotics and Artificial Intelligence**  
**Project Proposal | Autumn 2026**

### 1. Problem Statement and Motivation

Autonomous agents operating under uncertainty must continually decide whether to act on their current observations or gather additional information before committing to a decision. This tradeoff is fundamental to robotics, where an agent may need to choose between navigating with an uncertain state estimate and collecting additional sensor measurements. Similar decisions arise in scientific image analysis, medical decision support, and multimodal AI systems. Although an agent's final decision can be evaluated for correctness, its underlying preferences for information gathering, efficiency, and task completion are often difficult to characterize.

This project investigates whether **Bayesian Inverse Reinforcement Learning (BIRL)** can recover interpretable reward preferences from an agent's sequential decision-making behavior and whether uncertainty in these inferred preferences provides insight into decision failures.

We use Vision-Language Models (VLMs) as an experimental testbed, focusing on visual reasoning tasks in which a model must choose between inspecting visual evidence, performing intermediate reasoning, and committing to an answer. By observing these decision trajectories, we aim to infer how the model implicitly trades off information acquisition, reasoning cost, and answer commitment.

The broader objective is to develop a framework for **behavioral auditing of agents operating under uncertain observations**, with potential applications to embodied AI, autonomous robotics, and other systems that must decide when additional evidence is worth acquiring.

### 2. Relation to Course Material

The project directly connects to several topics covered in TTIC 31170:

- **Markov Decision Processes (MDPs):** We model visual reasoning as a finite-horizon sequential decision problem, with states representing the agent's accumulated evidence, actions corresponding to information gathering or reasoning operations, and rewards capturing latent behavioral preferences.
- **Inverse Reinforcement Learning:** Rather than specifying the agent's reward function, we infer it from observed decision trajectories using Maximum-Entropy IRL.
- **Bayesian Inference:** We extend point-estimate reward recovery to Bayesian IRL, maintaining a posterior distribution over reward parameters to characterize ambiguity in the inferred preferences.
- **Planning Under Uncertainty:** The decision to gather additional information or act using current observations reflects a general planning problem in partially observed environments.

### 3. Proposed Approach

We will construct a small, controlled visual reasoning environment in which a VLM selects from a discrete set of actions, such as **Inspect**, **Reason**, and **Answer**. Each action updates a structured reasoning state, allowing us to observe sequential decision trajectories rather than relying solely on free-form generated explanations.

We formulate the environment as a finite-horizon MDP, with states $s_t$ representing the image-question context and currently available evidence, actions $a_t$ representing reasoning or information-gathering operations, and transitions $T(s_{t+1} \mid s_t, a_t)$ describing how the evidence state changes.

The reward function is parameterized as a linear combination of interpretable features:

$$
R_\theta(s_t, a_t, s_{t+1}) = \theta^\top \phi(s_t, a_t, s_{t+1})
$$

Features will capture information acquisition, reasoning cost, and answer commitment, with an optional measure of evidence consistency. We will first implement Maximum-Entropy IRL to infer reward weights from observed trajectories. We will then introduce a Bayesian formulation:

$$
p(\theta \mid D) \propto p(D \mid \theta)\, p(\theta)
$$

where $D$ denotes observed reasoning trajectories and $p(\theta)$ is a prior over reward parameters. A low-dimensional reward model and finite planning horizon will allow tractable posterior approximation.

The experimental investigation will proceed in three stages:

1. **Reward recovery:** Validate the IRL implementation in a synthetic sequential environment with known reward parameters, then infer reward preferences from VLM-generated decision trajectories.
2. **Behavioral comparison:** Compare inferred reward preferences across successful and unsuccessful visual reasoning episodes, as well as image-conditioned and reduced-visual-information settings.
3. **Uncertainty analysis:** Investigate whether posterior uncertainty over reward preferences is associated with incorrect decisions, premature answer commitment, or insufficient information gathering.

This approach is appropriate because IRL provides a principled framework for inferring behavioral objectives from sequential decisions, while Bayesian inference explicitly represents uncertainty arising from limited or ambiguous demonstrations.

### 4. Anticipated Technical Challenges

The primary challenge is designing a compact state and action space that captures meaningful information-gathering decisions while remaining computationally tractable. We will address this by restricting the environment to a small number of discrete reasoning actions and a short planning horizon.

A second challenge is **reward identifiability**: multiple reward functions may explain similar trajectories. Bayesian inference will allow us to characterize this ambiguity, while synthetic experiments with known rewards will help validate the inference procedure.

Finally, VLM-generated actions may reflect prompting effects or limitations of the controlled environment. We will use consistent action definitions, fixed interaction protocols, and controlled comparisons to assess the stability of the inferred preferences.

### 5. Evaluation Metrics

We will evaluate success along three dimensions:

- **Reward recovery:** Parameter estimation error and posterior credible-interval coverage on synthetic demonstrations with known reward functions.
- **Behavioral modeling:** Held-out trajectory log-likelihood and the ability of inferred rewards to predict observed action choices.
- **Diagnostic utility:** Association between inferred reward preferences or posterior uncertainty and reasoning failures, measured using classification metrics such as AUROC, alongside comparisons with simpler behavioral baselines.

We will also examine whether reward estimates change meaningfully when visual evidence is restricted or when task difficulty increases.

### 6. Datasets, Software, and Hardware

We plan to use a subset of **ScienceQA**, emphasizing image-dependent multiple-choice questions, together with an open-weight VLM such as LLaVA. A small controlled environment will provide structured information-gathering actions and recorded trajectories. Synthetic demonstrations will be generated for reward-recovery validation.

Implementation will use Python, PyTorch, Hugging Face Transformers, NumPy, and SciPy, with Pyro considered for Bayesian posterior inference. GPU resources will be used for VLM inference, while the small-state IRL experiments can be run with comparatively modest computational requirements.

### 7. Project Plan

1. Define the controlled reasoning environment, implement action/state representations and collect pilot VLM trajectories.
2. Implement Maximum-Entropy IRL and validate reward recovery on synthetic demonstrations.
3. Infer reward preferences from VLM trajectories and implement Bayesian posterior estimation.
4. Evaluate behavioral differences, posterior uncertainty and associations with reasoning failures.
5. Conduct targeted ablations, consolidate results and prepare the final report and presentation.


If Bayesian inference or the full reasoning environment proves computationally demanding, we will reduce the state space and prioritize validated Maximum-Entropy IRL with a focused behavioral analysis. This will preserve the project's central investigation of information-gathering preferences under uncertainty.

### References

Ziebart, B. D., et al. (2008). *Maximum Entropy Inverse Reinforcement Learning.* AAAI.

Ramachandran, D., & Amir, E. (2007). *Bayesian Inverse Reinforcement Learning.* IJCAI.

Lu, P., et al. (2022). *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering.* NeurIPS.
