# Introduction to Fine-Tuning Large Language Models

Welcome to this comprehensive course on fine-tuning Large Language Models (LLMs). This course is designed to take you from the basics of model training and transfer learning to advanced techniques in fine-tuning LLMs, vision-language models, and more. Whether you're a beginner or an experienced practitioner, you'll gain hands-on experience with the latest frameworks and tools.

Throughout the course, we'll cover theoretical concepts, practical implementations, and real-world applications. By the end, you'll be equipped to fine-tune models for various tasks, understand when to use fine-tuning versus other approaches like RAG or AI agents, and deploy your fine-tuned models effectively.

## Course Content Structure

- [ ] **01: Introduction to Fine-Tuning in Artificial Intelligence (AI)**
  - [ ] What is model training? (Machine Learning (ML), Deep Learning (DL), Computer Vision (CV), Natural Language Processing (NLP), Generative AI (GenAI))
  - [ ] Transfer learning (live demo with Convolutional Neural Network (CNN))
  - [ ] What is pretraining, fine-tuning, and why it matters
  - [ ] Pros and cons of fine-tuning
  - [ ] Overview of top Large Language Model (LLM) fine-tuning frameworks: Hugging Face (HF) Transformer Reinforcement Learning (TRL), Unsloth, LlamaFactory, Axolotl
  - [ ] Important research papers
  - [ ] Tips before fine-tuning

- [ ] **02: Fine-Tuning vs Retrieval-Augmented Generation (RAG) vs Artificial Intelligence (AI) Agents - What to Use When?**
  - [ ] Definitions: fine-tuning, transfer learning, Retrieval-Augmented Generation (RAG), agents
  - [ ] Comparison table
  - [ ] Use cases and when to use what
  - [ ] Industry examples

- [ ] **03: Fine-Tuning in Deep Learning**
  - [ ] Convolutional Neural Network (CNN) example end-to-end (image classification)
  - [ ] How feature extraction and fine-tuning work

- [ ] **04: Why Fine-Tuning Is Not Possible in Recurrent Neural Network (RNN) / Long Short-Term Memory (LSTM)?**
  - [ ] Limitations of older architectures (Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM))
  - [ ] Compare with transformer-based models

- [ ] **05: Hugging Face vs LangChain | Full Hugging Face (HF) Tutorial**
  - [ ] Hugging Face (HF) installation and environment setup
  - [ ] Using Hugging Face (HF) Application Programming Interfaces (APIs) vs offline downloads

- [ ] **06: Fine-Tuning Classical Language Models**
  - [ ] Bidirectional Encoder Representations from Transformers (BERT) fine-tuning (text classification / Question Answering (QA))
  - [ ] Text-To-Text Transfer Transformer (T5) fine-tuning (text-to-text tasks)

- [ ] **07: Knowledge Distillation in Large Language Models (LLMs)**
  - [ ] What is knowledge distillation?
  - [ ] Example: Bidirectional Encoder Representations from Transformers (BERT) -> DistilBERT
  - [ ] Combine with fine-tuning for better efficiency

- [ ] **08: Large Language Model (LLM) Quantization Explained**
  - [ ] What is quantization?
  - [ ] Weight quantization techniques
  - [ ] GGUF (GGML Unified Format), GGML (Georgi Gerganov Machine Learning), GPTQ (Generative Pre-trained Transformer Quantization), AWQ (Activation-aware Weight Quantization), INT4/INT8 (4-bit/8-bit Integer Quantization) demos
  - [ ] Why quantization matters

- [ ] **09: Fine-Tuning Large Language Models (Llama, Mistral, Gemma, Phi-3)**
  - [ ] One-command fine-tuning demo
  - [ ] Parameter-Efficient Fine-Tuning (PEFT) techniques: Low-Rank Adaptation (LoRA), Quantized Low-Rank Adaptation (QLoRA), Weight-Decomposed Low-Rank Adaptation (DoRA), Representation Fine-Tuning (ReFT)
  - [ ] Structured output fine-tuning (Quantized Low-Rank Adaptation (QLoRA))
  - [ ] Chat-based fine-tuning (OpenAssistant-like)
  - [ ] Full fine-tuning vs parameter-efficient fine-tuning
  - [ ] Dataset prep: Wiki dataset, FinWeb
  - [ ] Tools: Axolotl, Apple MLX, Unsloth
  - [ ] Post-finetune deployment (Ollama integration)

- [ ] **10: Application Programming Interface (API)-Based Model Fine-Tuning (GPT-4o, Gemini, etc.)**
  - [ ] OpenAI fine-tuning walkthrough
  - [ ] Distillation as an alternative for API models
  - [ ] Gemini fine-tune insights
  - [ ] Generative Pre-trained Transformer 4o (GPT-4o) use case fine-tuning

- [ ] **11: Best Frameworks for Large Language Model (LLM) Fine-Tuning**
  - [ ] Hands-on: LlamaFactory, Unsloth
  - [ ] Minimal code, max performance
  - [ ] Compare speed, memory, and flexibility

- [ ] **12: Fine-Tuning Vision-Language Models (VLMs)**
  - [ ] What are Vision-Language Models (VLMs)? (Vision Transformer (ViT), Florence2, Qwen2, LLaGemma)
  - [ ] Fine-tuning Vision-Language Models (VLMs) with LlamaFactory
  - [ ] Upload adapters and models to Hugging Face Hub

- [ ] **13: Reinforcement Learning from Human Feedback (RLHF)**
  - [ ] Proximal Policy Optimization (PPO) vs Direct Preference Optimization (DPO)
  - [ ] Real examples and when to use what
  - [ ] How Reinforcement Learning from Human Feedback (RLHF) fits into the fine-tuning pipeline

- [ ] **14: Embedding Fine-Tuning Deep Dive**
  - [ ] Embedding vs fine-tuning: conceptual and practical
  - [ ] Supervised Fine-Tuning (SFT) vs Unsupervised Fine-Tuning (USFT) (fine-tuning variants)
  - [ ] Embedding for retrieval and semantic search
  - [ ] When to fine-tune embeddings vs models