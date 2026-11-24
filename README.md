# Inferring Information-Gathering Preferences in Multimodal Agents via Bayesian Inverse Reinforcement Learning

**TTIC 31170 - Planning, Learning, and Estimation for Robotics and Artificial Intelligence**

**Project Proposal | Autumn 2026**

## 1. Problem Statement and Motivation

Agents operating under uncertain observations must decide whether to act using their current evidence or pay a cost to gather more information. This tradeoff is fundamental to robotics: a system may need to choose between acting from an uncertain state estimate and collecting another sensor measurement. The same structure appears in high-resolution visual question answering, where a model can answer from a coarse view or inspect a more informative region before committing.

This project investigates whether **Bayesian Inverse Reinforcement Learning (BIRL)** can recover interpretable preferences from a vision-language model's sequential information-gathering behavior. The primary question is:

> When a multimodal model answers incorrectly, did it stop before collecting useful evidence, or did it collect evidence but remain uncertain or miscalibrated?

The project treats this as a controlled **stopping-under-progressive-evidence** problem. A model repeatedly chooses between **Inspect**, which reveals a more informative observation at a cost, and **Answer**, which terminates the episode. By observing these trajectories, we aim to infer how the model trades off information acquisition, inspection cost, and answer utility.

The broader objective is a framework for behavioral auditing of agents that must decide when additional evidence is worth acquiring. The conclusions will be behavioral rather than mechanistic: an inferred reward posterior describes objectives consistent with observed choices under the assumed environment and policy model, not the model's internal reward function.

## 2. Dataset and Experimental Setting

### Primary pilot dataset: V*Bench

The primary dataset will be **V*Bench**, a visual question-answering benchmark designed around high-resolution, visually crowded images in which the relevant evidence is often a small object or detail. It contains **191 multiple-choice questions** over images with an average resolution of approximately **2246 x 1582**:

- **Attribute recognition:** 115 questions with four answer choices, testing attributes such as object color or material.
- **Spatial relationship reasoning:** 76 questions with two answer choices, testing the relationship between two objects.

V*Bench is a good pilot environment because the need for additional evidence arises from genuine visual scale and crowding rather than an artificial blur applied to an otherwise conventional question-answering dataset. The released annotations include target-object names and manually labeled bounding boxes. These annotations make it possible to construct reproducible evidence levels and, for spatial questions, to ensure that both relevant objects remain inside the revealed crop.

An official V*Bench implementation is available in **Inspect Evals**. It provides dataset loading, standard model execution, logging, and multiple-choice scoring. The existing implementation is a one-step accuracy evaluation and does not implement the V* visual-search algorithm or the sequential Inspect/Answer environment proposed here. We will therefore reuse its dataset and scoring conventions while adding a custom sequential task or solver.

### Conditional scale-up: MME-RealWorld

V*Bench was selected as it is intentionally small, so it will be treated as a pilot and method-development benchmark rather than the sole source of high-powered failure analysis. Reason being that repeated rollouts on the same question are correlated which can obscure reward recivery.

If the V*Bench pilot produces meaningful variation in stopping decisions, we will evaluate a **1,500-2,500 question subset of MME-RealWorld**, prioritizing autonomous-driving, monitoring, and remote-sensing tasks. 

MME-RealWorld contains 13,366 high-resolution images and 29,429 human-annotated question-answer pairs across 43 tasks. This scale-up is conditional on defining reproducible progressive observations for the selected subset. If suitable localization annotations are unavailable, we will use fixed image tiles or manually annotate a smaller evaluation subset rather than silently introduce oracle information.

## 3. Sequential Environment

Each episode contains one image, a question, its answer choices, and three progressively informative visual observations.

| Evidence level | Observation |
|---|---|
| **Level 0** | Downsampled full image and the question with answer choices |
| **Level 1** | Coarse, context-preserving crop around the annotated target region at moderate resolution |
| **Level 2** | Tighter target-region crop at native or near-native resolution |

For spatial-relation questions, Levels 1 and 2 will be constructed from the union of the two target boxes so that both objects remain visible. Crop expansion factors and output resolutions will be fixed before trajectory collection.

At Levels 0 and 1, the model chooses one of:

- `INSPECT`: reveal the next evidence level.
- `ANSWER_<choice>`: submit one of the available answers and terminate.

At Level 2, the model must answer. The prompt will define the available actions but will not force the model to inspect. This freedom is essential: trajectories reveal little about preferences if every episode follows a prescribed number of inspections.

Using annotated target crops gives the agent oracle localization. Consequently, the main experiment measures **when to stop inspecting**, not **where to look**. A non-oracle localization or fixed-tile ablation may be added if time permits, but it is not required for the minimum viable experiment.

## 4. State, Features, and Reward Model

The environment is a short, finite-horizon decision process with deterministic evidence transitions. A compact state representation will include:

- current evidence level;
- task type and number of answer choices;
- normalized predictive entropy over answer choices;
- top-one probability and probability margin;
- inspection history; and
- image or question difficulty indicators available before termination.

For a question with \(K\) answer choices and predictive probabilities \(p_1,\ldots,p_K\), uncertainty will be represented by normalized entropy:

$$
\bar{H}(p) = \frac{-\sum_{k=1}^{K}p_k\log p_k}{\log K}.
$$

This normalization is necessary because attribute questions have four choices while spatial questions have two. It maps maximum entropy to one in both tasks and makes uncertainty-related features comparable.

The reward remains a low-dimensional linear function:

$$
R_\theta(s_t,a_t,s_{t+1}) =
\theta_1 f_{\text{information gain}} +
\theta_2 f_{\text{inspection cost}} +
\theta_3 f_{\text{answer utility}}.
$$

The initial feature definitions will be:

- **Information gain:** reduction in normalized answer entropy after an inspection.
- **Inspection cost:** an indicator or fixed negative cost for requesting another observation.
- **Answer utility:** terminal utility based on answer correctness, with sensitivity analyses for alternative correct/incorrect reward scales.

We will avoid adding many correlated reward features until synthetic recovery experiments show that their weights are identifiable. The model's action likelihood will be represented with a soft or Boltzmann-rational policy derived from the finite-horizon action values. Maximum-Entropy IRL will provide a point-estimate baseline, followed by a Bayesian posterior:

$$
p(\theta \mid D) \propto p(D \mid \theta)p(\theta),
$$

where \(D\) contains complete state-action trajectories. Posterior inference will use a low-dimensional sampler or grid-based approximation when practical, with Pyro considered only if it simplifies a validated implementation.

## 5. Calibration as a Separate Measurement

Predictive entropy is both an observed behavioral signal and a potential reward feature. Therefore, the reward posterior alone cannot determine whether a failure was caused by miscalibrated uncertainty or by under-weighting information gathering.

Calibration will be measured separately at each evidence level using held-out answer probabilities and correctness. Metrics will include:

- expected calibration error with reported binning choices;
- Brier score;
- negative log-likelihood; and
- accuracy and confidence as functions of evidence level.

The inferred reward posterior will then be interpreted alongside these calibration measurements. For example, premature commitment with well-calibrated low confidence differs from premature commitment accompanied by unjustified high confidence. This comparison is an analysis of two measured phenomena, not a claim that BIRL alone causally separates them.

## 6. Experimental Plan

### Stage 1: Environment and descriptive baselines

Implement the three evidence levels, structured actions, trajectory logging, and the standard accuracy scorer. Before fitting IRL, measure whether the environment creates a meaningful information gradient using:

- question-only and text-only accuracy;
- answer accuracy at each forced evidence level;
- forced immediate-answer and forced full-inspection policies;
- entropy, confidence, and calibration at each level; and
- the fraction of episodes ending at each inspection depth.

The pilot will begin with 20-30 questions for debugging and then expand to all 191 V*Bench questions if the observation transformations behave as intended.

### Stage 2: Synthetic reward recovery

Construct synthetic agents with known reward parameters in the same finite-horizon environment. Validate whether Maximum-Entropy IRL and BIRL recover these parameters under different trajectory counts, policy temperatures, and feature correlations. This stage is required before interpreting reward weights inferred from VLM behavior.

### Stage 3: VLM trajectory collection

Use an open-weight VLM that permits access to answer-token logits or probabilities. For each question, collect structured Inspect/Answer trajectories under a fixed prompt and decoding protocol. Multiple stochastic rollouts may be used to estimate within-question variability, but train/test splitting and uncertainty estimates will operate at the **question level**, not treat rollouts as independent samples.

### Stage 4: Behavioral inference and evaluation

Fit reward models to the collected trajectories and evaluate:

- held-out action and trajectory log-likelihood;
- prediction of inspection depth and stopping decisions;
- posterior recovery and credible-interval coverage in synthetic experiments;
- differences in inferred preferences between correct and incorrect episodes;
- accuracy-cost tradeoffs relative to simple confidence-threshold policies; and
- association between inferred preferences, calibration, and failures.

Failure-prediction AUROC will be considered exploratory on V*Bench because 191 questions may yield relatively few errors. We will report bootstrap confidence intervals resampled by question and avoid presenting correlated rollouts as additional independent evidence. If the pilot is successful, the MME-RealWorld scale-up will provide a more appropriate sample size for failure-prediction claims.

## 7. Relation to Course Material

- **Markov Decision Processes:** The progressive-evidence task is a finite-horizon decision process with known transitions and terminal answer actions.
- **Inverse Reinforcement Learning:** Reward parameters are inferred from observed stopping and inspection trajectories rather than specified as the behavioral explanation.
- **Bayesian Inference:** A posterior over reward parameters represents ambiguity arising from limited or behaviorally similar demonstrations.
- **Planning Under Uncertainty:** Inspecting trades an immediate cost for a potentially better-informed terminal decision.
- **Partially Observed Decision-Making:** Evidence levels approximate changes in the agent's information state while the complete high-resolution scene remains latent at early steps.

## 8. Anticipated Challenges and Mitigations

### Reward identifiability

Information gain, confidence, correctness, and inspection depth may be correlated. We will use a small reward feature set, synthetic recovery experiments, posterior diagnostics, and feature ablations before assigning behavioral interpretations to individual weights.

### Prompt-induced behavior

Structured action prompts can influence stopping decisions. We will freeze the prompt before the main collection, report it in full, and run a limited prompt-sensitivity check.

### Oracle localization

Annotated target crops make the evidence sequence reproducible but remove the search problem. We will explicitly restrict our main claim to stopping under progressive evidence. A non-oracle crop or localization ablation will be treated as an extension.

### Limited independent sample size

V*Bench's 191 questions are sufficient for a controlled pilot but may not support precise subgroup or AUROC estimates. Question-level splits, clustered or hierarchical bootstrap procedures, and the conditional MME-RealWorld scale-up will address this limitation.

### 7. Anticipated Timeline

1. Define the controlled reasoning environment, implement action/state representations, and collect pilot VLM trajectories. 
2. Implement Maximum-Entropy IRL and validate reward recovery on synthetic demonstrations. 
3. Infer reward preferences from VLM trajectories and implement Bayesian posterior estimation. 
4. Evaluate behavioral differences, posterior uncertainty, and associations with reasoning failures. 
5. Conduct targeted ablations, consolidate results, and prepare the final report and presentation (If available)

If Bayesian inference proves unreliable or computationally excessive, the project will prioritize validated Maximum-Entropy IRL, synthetic recovery, calibration analysis, and the behavioral stopping experiment. This preserves the central investigation without overstating what can be learned from a small dataset.

## References

- Wu, P., & Xie, S. (2024). *V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs. CVPR. [Paper](https://arxiv.org/abs/2312.14135) | [Code and annotations](https://github.com/penghao-wu/vstar)
- UK AI Security Institute and contributors. *V*Bench: A Visual QA Benchmark with Detailed High-resolution Images. [Inspect Evals implementation](https://ukgovernmentbeis.github.io/inspect_evals/evals/vstar_bench/)
- Zhang, Y.-F., et al. (2024). *MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios that are Difficult for Humans?* [Project page](https://mme-realworld.github.io/home_page.html)
- Ziebart, B. D., et al. (2008). *Maximum Entropy Inverse Reinforcement Learning.* AAAI.
- Ramachandran, D., & Amir, E. (2007). *Bayesian Inverse Reinforcement Learning.* IJCAI.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML.
