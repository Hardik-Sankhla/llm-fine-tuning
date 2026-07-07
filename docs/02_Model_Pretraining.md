# 02 - Model Pretraining and Foundations

This section is a complete, structured guide to model pretraining and how it connects to modern LLM fine-tuning.

## 1. Where Pretraining Fits in the AI Pipeline

A typical model-development lifecycle:
1. Data collection and ingestion
2. Data analysis and quality checks
3. Data preprocessing and tokenization
4. Model training (pretraining and/or fine-tuning)
5. Evaluation and error analysis
6. Iteration and deployment

For modern language systems, pretraining builds the base model, and fine-tuning/alignment specialize it for real tasks.

## 2. Learning Paradigms in One View

- Supervised learning: train with input-label pairs.
- Unsupervised learning: discover structure without explicit labels.
- Self-supervised learning: generate labels from the data itself (most LLM pretraining).
- Reinforcement learning: optimize behavior through rewards.

Note: LLM pretraining is often called "unsupervised," but technically it is usually self-supervised.

## 3. Model Families and Historical Context

### 3.1 Classical ML

Common supervised models:
- Linear Regression
- Logistic Regression
- SVM
- Decision Tree
- Random Forest
- Gradient boosting methods (XGBoost, LightGBM)

Common unsupervised methods:
- K-Means
- Hierarchical clustering
- DBSCAN

### 3.2 Deep Learning Progression

- ANN: general tabular/regression/classification tasks.
- CNN: grid-structured inputs (images, video frames).
- RNN/LSTM/GRU: sequence modeling before Transformers became dominant.
- Transformer: attention-first architecture powering modern LLMs.

## 4. Pretraining in Computer Vision vs LLMs

### 4.1 Computer Vision Pretraining

In vision, pretraining typically means training a CNN/ViT backbone on a large dataset (for example ImageNet) and transferring those learned features.

Common pretrained backbones:
- VGG16/VGG19
- ResNet50/ResNet101
- InceptionV3
- MobileNet
- EfficientNet

Feature hierarchy:
- Early layers: edges, corners, textures
- Middle layers: motifs and parts
- Deep layers: semantic object parts and classes

### 4.2 LLM Pretraining

LLM pretraining teaches language patterns from massive corpora.

Typical flow:
1. Collect raw corpora (web, books, code, docs)
2. Clean and deduplicate
3. Tokenize text
4. Train Transformer with a self-supervised objective
5. Validate with held-out data (loss/perplexity)
6. Save pretrained weights

## 5. Core Pretraining Objectives

### 5.1 Causal Language Modeling (CLM)

- Predict next token from left context.
- Used by GPT-style decoder-only models.

### 5.2 Masked Language Modeling (MLM)

- Predict masked tokens using both left and right context.
- Used by BERT-style encoder models.

### 5.3 Span Corruption / Text-to-Text

- Predict missing spans or reformulate all tasks as text-to-text.
- Used in T5-like objectives.

## 6. Why Transformers Replaced RNN/LSTM for Scale

Compared with recurrence-based models, Transformers:
- Train more efficiently with parallelism.
- Handle long-range dependencies better in practice.
- Scale effectively with compute, data, and parameters.

Landmark context:
- "Attention Is All You Need" introduced the Transformer architecture.
- BERT popularized bidirectional pretraining for NLU tasks.
- GPT-style CLM scaled to strong few-shot performance in GPT-3.

## 7. From Pretraining to Fine-Tuning

Pretraining creates a foundation model; fine-tuning adapts it to target tasks.

Common adaptation strategies:
- Feature extraction: freeze base, train lightweight head.
- Partial fine-tuning: unfreeze selected layers.
- Full fine-tuning: update most/all parameters.
- PEFT: train adapters/low-rank updates (LoRA/QLoRA) with lower cost.

Why transfer learning works:
- Reuses generalized language features.
- Requires less labeled data for downstream tasks.
- Cuts compute and training time compared with training from scratch.

## 8. Modern 3-Stage LLM Training Stack

### Stage 1: Pretraining

Goal: learn broad language/world structure from large unlabeled corpora.

Typical metrics:
- Training/validation loss
- Perplexity
- Token accuracy trends

### Stage 2: Supervised Fine-Tuning (SFT)

Goal: improve instruction following and task behavior using labeled examples.

Typical tasks:
- Classification
- Summarization
- Question answering
- Dialogue
- Structured extraction

### Stage 3: Alignment / Preference Optimization

Goal: align outputs with human preferences and safety policies.

Common methods:
- RLHF (reward model + policy optimization such as PPO)
- Direct preference methods such as DPO

In practice, many teams now prefer simpler offline preference methods when they provide comparable quality with lower complexity.

## 9. Practical Framework Mapping

### Hugging Face Transformers

Use for:
- Loading pretrained models/tokenizers
- Standard training and evaluation pipelines

### PEFT

Use for:
- Parameter-efficient adaptation (LoRA, prompt tuning, etc.)
- Lower VRAM and storage footprint for fine-tuning

### TRL

Use for:
- SFT, reward modeling, DPO/PPO-style post-training
- Experimentation with preference optimization workflows

## 10. Data Quality Checklist Before Pretraining/Fine-Tuning

- Remove duplicates and near-duplicates
- Filter toxic and policy-violating content
- Reduce leakage from eval/test sets
- Track domain balance and language distribution
- Validate token length distributions
- Document dataset licenses and provenance

## 11. Evaluation Checklist

- Intrinsic: loss, perplexity, calibration trends
- Task-level: benchmark metrics (accuracy, F1, Rouge, exact match)
- Human eval: helpfulness, correctness, safety
- Robustness: out-of-domain and long-context behavior
- Failure analysis: hallucination and formatting errors

## 12. Common Pitfalls and Fixes

- Overfitting small task data:
  - Use PEFT, lower learning rate, early stopping, stronger validation.
- Catastrophic forgetting:
  - Mix instruction data, reduce aggressive updates, consider adapters.
- High compute cost:
  - Use QLoRA, gradient checkpointing, mixed precision, smaller context windows.
- Poor instruction following:
  - Improve prompt format consistency and high-quality SFT examples.

## 13. Quick Decision Guide

- Need domain facts updated frequently: start with RAG.
- Need style/format/behavior shift: use SFT or PEFT fine-tuning.
- Need policy preference shaping: add DPO/RLHF layer.
- Need minimal infra and fast iteration: use PEFT first.

## 14. Key Takeaways

- Pretraining builds general intelligence priors; fine-tuning specializes behavior.
- Modern LLM systems are built as staged pipelines, not one-shot training.
- Data quality and evaluation discipline matter more than raw model size in many applied settings.
- PEFT and preference optimization make high-quality adaptation practical on limited hardware.

## 15. References and Suggested Reading

- Vaswani et al., Attention Is All You Need, arXiv:1706.03762
  - https://arxiv.org/abs/1706.03762
- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, arXiv:1810.04805
  - https://arxiv.org/abs/1810.04805
- Brown et al., Language Models are Few-Shot Learners, arXiv:2005.14165
  - https://arxiv.org/abs/2005.14165
- Hugging Face PEFT docs
  - https://huggingface.co/docs/peft/en/index
- Hugging Face TRL docs
  - https://huggingface.co/docs/trl/en/index

---

Next section recommendation: Fine-Tuning vs RAG vs Agents.
