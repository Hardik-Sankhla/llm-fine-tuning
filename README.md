# LLM Fine-Tuning Toolkit

A comprehensive toolkit for fine-tuning Large Language Models using Hugging Face Transformers and Parameter-Efficient Fine-Tuning (PEFT) techniques.

## Features

- **Parameter-Efficient Fine-Tuning**: Support for LoRA and other PEFT methods
- **Flexible Model Support**: Compatible with any Hugging Face model
- **Dataset Integration**: Easy loading from Hugging Face datasets
- **Training Optimization**: Automatic mixed precision and gradient accumulation
- **Evaluation**: Built-in evaluation during training
- **Experiment Tracking**: Integration with Weights & Biases

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

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
