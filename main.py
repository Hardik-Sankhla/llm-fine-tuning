#!/usr/bin/env python3
"""
LLM Fine-Tuning Toolkit

This script provides a basic implementation for fine-tuning Large Language Models
using Hugging Face Transformers and PEFT (Parameter-Efficient Fine-Tuning).
"""

import argparse
import logging
import os
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
import wandb


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer, max_length: int = 512):
    """Tokenize the input text."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def prepare_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    test_size: float = 0.1,
):
    """Load and prepare the dataset."""
    dataset = load_dataset(dataset_name, split="train")

    # Tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Split
    split_dataset = tokenized_dataset.train_test_split(test_size=test_size)
    return split_dataset


def load_model(model_name: str, use_peft: bool = True):
    """Load the model with optional PEFT."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if use_peft:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    return model


def create_data_collator(tokenizer: AutoTokenizer):
    """Create data collator for language modeling."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )


def create_training_args(output_dir: str, num_epochs: int = 3):
    """Create training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )


def main(
    model_name: str,
    dataset_name: str,
    output_dir: str,
    num_epochs: int = 3,
    max_length: int = 512,
    use_peft: bool = True,
):
    """Main fine-tuning function."""
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting LLM fine-tuning...")

    # Initialize wandb
    wandb.init(project="llm-fine-tuning", name=f"{model_name.split('/')[-1]}-{dataset_name}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)

    # Prepare dataset
    logger.info(f"Preparing dataset: {dataset_name}")
    dataset = prepare_dataset(dataset_name, tokenizer, max_length)

    # Load model
    logger.info(f"Loading model: {model_name}")
    model = load_model(model_name, use_peft)

    # Create data collator
    data_collator = create_data_collator(tokenizer)

    # Create training args
    training_args = create_training_args(output_dir, num_epochs)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Fine-tuning completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Large Language Model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/DialoGPT-small",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for the fine-tuned model",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--use_peft",
        action="store_true",
        default=True,
        help="Use PEFT for parameter-efficient fine-tuning",
    )

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        use_peft=args.use_peft,
    )
