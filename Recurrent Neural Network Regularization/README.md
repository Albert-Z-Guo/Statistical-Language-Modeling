# Recurrent Neural Network Regularization
This repository contains an adaptation of the [TensorFlow implementation](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb) of Wojciech Zaremba's [Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329) for Number Token Perplexity Measurement. All results were obtained with an NVIDIA's [GeForce RTX 2080 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2080-ti/) Graphics Card.

## Number Token Definitions
- rounds = [10, 20, ..., 90, 100, 200, ..., 900]
- days = [1, 2, ..., 31]
- years = [1000, 1001, ..., 2020]

## Implementation Details
This implementation uses the same setting as the one mentioned in the paper, except the following changes:
1. The corpora used for training, validation, and testing is now [Wikitext-2](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) instead of [Penn Tree Bank](https://catalog.ldc.upenn.edu/LDC99T42) (PTB).
2. The vocabulary size is increased from 10000 to about 27250 (since Wikitext-2's vocabulary size is larger than PTB's vocabulary size which is exactly 10000) for each word with frequency > 3.
3. For words in valid and test data but not in training data, they are treated as 'UNK' instead of not reading them.
4. Retrieval of an additional `prob` ndarray ([batch_size, num_steps, vocab_size]) is enabled by adding a softmax function in computation graph, @property, and relevant export_ops and import_ops support.
5. By processing training, validation, and test data, number tokens' indices are extracted for perplexity calculation via `prob` ndarray.

## Environment Setup
To install all libraries/dependencies used in this project, run
```bash
pip3 install -r requirement.txt
```

## Performance Evaluation
### Experiment
To experiment a model, run:
```bash
python3 ptb_word.py --data_path=data --model=model_choice --save_path=token_class_name --token_class=token_class_name --num_gpus=num_gpus --run_mode=run_mode
```

- `data_path`      is the data path; default is `data`
- `model_choice`   is the model to choose from: `small`, `medium`, or `large`; default is `small`
- `save_path`      is the path to save model; default is `rounds` which is the default number token class
- `token_class`    is the number token class to choose from: `rounds`, `days`, or `years`; default is `rounds`
- `num_gpus`       is the number of gpus available; default is `1`
- `run_mode`       is the low level implementation of LSTM cell to choose from `CUDNN`, `BASIC`, or `BLOCK`, representing cudnn_lstm, basic_lstm, and lstm_block_cell classes; default is `CUDNN`

### Results
| Number Token Class | Test Perplexity|
|--------------------|----------------|
| rounds             | 798            |
| days               | 209            |
| years              | 398            |
