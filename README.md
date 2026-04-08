# LLM Fine-Tuning Toolkit

A comprehensive toolkit and learning track for fine-tuning Large Language Models (LLMs) using Hugging Face Transformers and Parameter-Efficient Fine-Tuning (PEFT) techniques.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Content Structure](#content-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Welcome to this comprehensive LLM fine-tuning toolkit and course outline. This repository is designed to take you from foundational model-training concepts to practical, advanced fine-tuning workflows, including distillation, quantization, RLHF, and VLM tuning.

The project includes practical scripts, configurable training flows, and structured learning content so you can quickly move from theory to real implementation.

## Features

- **Parameter-Efficient Fine-Tuning**: Support for LoRA and other PEFT methods
- **Flexible Model Support**: Compatible with Hugging Face models
- **Dataset Integration**: Easy loading from Hugging Face datasets
- **Training Optimization**: Automatic mixed precision and gradient accumulation
- **Evaluation Support**: Built-in evaluation during training
- **Experiment Tracking**: Integration with Weights & Biases

## Content Structure

- [ ] **01: Introduction to Fine-Tuning in Artificial Intelligence (AI)**

    - [ ] What is model training? (Machine Learning (ML), Deep Learning (DL), Computer Vision (CV), Natural Language Processing (NLP), and Generative AI (GenAI))
    - [ ] Transfer learning (live demo with Convolutional Neural Network (CNN))
    - [ ] What are pretraining and fine-tuning, and why do they matter?
    - [ ] Pros and cons of fine-tuning
    - [ ] Overview of top Large Language Model (LLM) fine-tuning frameworks: Hugging Face (HF) Transformer Reinforcement Learning (TRL), Unsloth, LlamaFactory, and Axolotl
    - [ ] Key research papers
    - [ ] Tips before starting fine-tuning

- [ ] **02: Fine-Tuning vs Retrieval-Augmented Generation (RAG) vs Artificial Intelligence (AI) Agents: What to Use and When**

    - [ ] Definitions: fine-tuning, transfer learning, Retrieval-Augmented Generation (RAG), and agents
    - [ ] Comparison table
    - [ ] Use cases and when to use each approach
    - [ ] Industry examples

- [ ] **03: Fine-Tuning in Deep Learning**

    - [ ] End-to-end Convolutional Neural Network (CNN) example (image classification)
    - [ ] How feature extraction and fine-tuning work

- [ ] **04: Why Fine-Tuning Is Not Effective for Recurrent Neural Networks (RNNs) and Long Short-Term Memory Networks (LSTMs)**

    - [ ] Limitations of older architectures (Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs))
    - [ ] Comparison with transformer-based models

- [ ] **05: Hugging Face vs LangChain | Full Hugging Face (HF) Tutorial**

    - [ ] Hugging Face (HF) installation and environment setup
    - [ ] Using Hugging Face (HF) Application Programming Interfaces (APIs) vs. offline downloads

- [ ] **06: Fine-Tuning Classical Language Models**

    - [ ] Bidirectional Encoder Representations from Transformers (BERT) fine-tuning (text classification / question answering (QA))
    - [ ] Text-to-Text Transfer Transformer (T5) fine-tuning (text-to-text tasks)

- [ ] **07: Knowledge Distillation in Large Language Models (LLMs)**

    - [ ] What is knowledge distillation?
    - [ ] Example: Bidirectional Encoder Representations from Transformers (BERT) to DistilBERT
    - [ ] Combining distillation with fine-tuning for better efficiency

- [ ] **08: Large Language Model (LLM) Quantization Explained**

    - [ ] What is quantization?
    - [ ] Weight quantization techniques
    - [ ] Demos: GGUF (GGML Unified Format), GGML (Georgi Gerganov Machine Learning), GPTQ (Generative Pre-trained Transformer Quantization), AWQ (Activation-aware Weight Quantization), and INT4/INT8 (4-bit/8-bit integer quantization)
    - [ ] Why quantization matters

- [ ] **09: Fine-Tuning Large Language Models (Llama, Mistral, Gemma, and Phi-3)**

    - [ ] One-command fine-tuning demo
    - [ ] Parameter-Efficient Fine-Tuning (PEFT) techniques: Low-Rank Adaptation (LoRA), Quantized Low-Rank Adaptation (QLoRA), Weight-Decomposed Low-Rank Adaptation (DoRA), Representation Fine-Tuning (ReFT)
    - [ ] Structured output fine-tuning (Quantized Low-Rank Adaptation (QLoRA))
    - [ ] Chat-based fine-tuning (OpenAssistant-like)
    - [ ] Full fine-tuning vs parameter-efficient fine-tuning
    - [ ] Dataset preparation: Wiki dataset and FinWeb
    - [ ] Tools: Axolotl, Apple MLX, Unsloth
    - [ ] Post-fine-tuning deployment (Ollama integration)

- [ ] **10: Application Programming Interface (API)-Based Model Fine-Tuning (GPT-4o, Gemini, etc.)**

    - [ ] OpenAI fine-tuning walkthrough
    - [ ] Distillation as an alternative for API models
    - [ ] Gemini fine-tuning insights
    - [ ] Generative Pre-trained Transformer 4o (GPT-4o) use-case fine-tuning

- [ ] **11: Best Frameworks for Large Language Model (LLM) Fine-Tuning**

    - [ ] Hands-on: LlamaFactory, Unsloth
    - [ ] Minimal code, max performance
    - [ ] Compare speed, memory, and flexibility

- [ ] **12: Fine-Tuning Vision-Language Models (VLMs)**

    - [ ] What are Vision-Language Models (VLMs)? (Vision Transformer (ViT), Florence2, Qwen2, LLaGemma)
    - [ ] Fine-tuning Vision-Language Models (VLMs) using LlamaFactory
    - [ ] Upload adapters and models to Hugging Face Hub

- [ ] **13: Reinforcement Learning from Human Feedback (RLHF)**

    - [ ] Proximal Policy Optimization (PPO) vs Direct Preference Optimization (DPO)
    - [ ] Real examples and when to use what
    - [ ] How Reinforcement Learning from Human Feedback (RLHF) fits into the fine-tuning pipeline

- [ ] **14: Embedding Fine-Tuning Deep Dive**

    - [ ] Embeddings vs. fine-tuning: conceptual and practical differences
    - [ ] Supervised Fine-Tuning (SFT) vs Unsupervised Fine-Tuning (USFT) (fine-tuning variants)
    - [ ] Embedding for retrieval and semantic search
    - [ ] When to fine-tune embeddings vs. models

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/llm-fine-tuning.git
cd llm-fine-tuning
```

2. Install dependencies:

```bash
pip install -e .
```

## Quick Start

Fine-tune a model with default settings:

```bash
python main.py --model_name microsoft/DialoGPT-small --dataset_name wikitext
```

## Usage

### Command Line Arguments

- `--model_name`: Hugging Face model name (default: microsoft/DialoGPT-small)
- `--dataset_name`: Hugging Face dataset name (default: wikitext)
- `--output_dir`: Output directory for the fine-tuned model (default: ./output)
- `--num_epochs`: Number of training epochs (default: 3)
- `--max_length`: Maximum sequence length (default: 512)
- `--use_peft`: Use PEFT for parameter-efficient fine-tuning (default: True)

### Example

```bash
python main.py \
    --model_name gpt2 \
    --dataset_name imdb \
    --output_dir ./fine-tuned-gpt2 \
    --num_epochs 5 \
    --max_length 1024
```

## Configuration

The toolkit uses sensible defaults but can be customized through command-line arguments. For advanced usage, modify the code directly.

## Requirements

- Python 3.13+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for faster training)

## Contributing

Contributions are welcome. Feel free to open issues and submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
