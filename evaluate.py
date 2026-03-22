#!/usr/bin/env python3
"""
Evaluation and Inference Script for Fine-Tuned LLMs
"""

import argparse
import logging
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_model_and_tokenizer(model_path: str):
    """Load the fine-tuned model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return model, tokenizer


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    num_return_sequences: int = 1,
) -> List[str]:
    """Generate text from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        # Remove the prompt from the generated text
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        generated_texts.append(text)

    return generated_texts


def evaluate_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_texts: List[str],
) -> float:
    """Calculate perplexity on evaluation texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            num_tokens = inputs["input_ids"].numel()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()


def main(
    model_path: str,
    prompt: Optional[str] = None,
    eval_file: Optional[str] = None,
    max_length: int = 100,
    temperature: float = 0.7,
    num_samples: int = 1,
):
    """Main evaluation function."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"Loading model from {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)

    if prompt:
        logger.info(f"Generating text for prompt: {prompt}")
        generated_texts = generate_text(
            model, tokenizer, prompt, max_length, temperature, num_samples
        )
        for i, text in enumerate(generated_texts):
            print(f"Generated text {i+1}: {text}")

    if eval_file:
        logger.info(f"Evaluating perplexity on {eval_file}")
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_texts = f.readlines()
        eval_texts = [text.strip() for text in eval_texts if text.strip()]

        perplexity = evaluate_perplexity(model, tokenizer, eval_texts)
        print(f"Perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and generate text with a fine-tuned LLM")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for text generation",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        help="File containing evaluation texts for perplexity calculation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum length for generated text",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for text generation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        prompt=args.prompt,
        eval_file=args.eval_file,
        max_length=args.max_length,
        temperature=args.temperature,
        num_samples=args.num_samples,
    )