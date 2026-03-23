import pytest
from unittest.mock import patch, MagicMock
from main import load_tokenizer, tokenize_function, prepare_dataset


def test_load_tokenizer():
    """Test tokenizer loading."""
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = '<eos>'

        tokenizer = load_tokenizer('test-model')

        mock_tokenizer.assert_called_once_with('test-model')
        assert tokenizer.pad_token == '<eos>'


def test_tokenize_function():
    """Test tokenization function."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}

    result = tokenize_function({'text': 'Hello world'}, mock_tokenizer, 512)

    mock_tokenizer.assert_called_once_with(
        'Hello world',
        truncation=True,
        max_length=512,
        padding=False
    )
    assert result == {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}


@patch('datasets.load_dataset')
def test_prepare_dataset(mock_load_dataset):
    """Test dataset preparation."""
    mock_dataset = MagicMock()
    mock_dataset.column_names = ['text']
    mock_dataset.map.return_value = mock_dataset
    mock_dataset.train_test_split.return_value = {'train': mock_dataset, 'test': mock_dataset}
    mock_load_dataset.return_value = mock_dataset

    with patch('main.load_tokenizer') as mock_load_tokenizer:
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        result = prepare_dataset('test-dataset', mock_tokenizer)

        mock_load_dataset.assert_called_once_with('test-dataset', split='train')
        assert 'train' in result
        assert 'test' in result