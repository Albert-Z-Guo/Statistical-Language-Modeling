# Statistical-Language-Modeling

## Deep Learning for Natural Language Processing
This repository contains modern implementation of the classic paper [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) by [Yoshua Bengio](https://en.wikipedia.org/wiki/Yoshua_Bengio) et al. in 2003 using [TensorFlow](https://www.tensorflow.org/). All results were obtained with an NVIDIA's GeForce RTX 2080 Ti Graphics Card.

This implementation uses the same setting as the one mentioned in the paper: the first 800,000 words for training, the following 200,000 words for validation, and the remaining words for testing. However, because the [Brown corpora](https://en.wikipedia.org/wiki/Brown_Corpus) used in this project has minor discrepancy in word numbers as compared to the Brown corpora used in the paper, the final vocabulary size of the word embedding use is slightly different from the one mentioned in the paper. In addition, [Wikitext-2 corpora](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) was used to evaluate MLP7's performance.

In addition, the weight decay used in this implementation leverages Tensorflow's built-in function `tf.contrib.opt.extend_with_decoupled_weight_decay` which includes biases, which are actually not included in the paper.

Another major difference worth to note is that Adam optimizer instead of stochastic gradient descent optimizer is used extensively in this project for faster convergence.

As for the gradually decreasing learning rate, this implementation uses the given `epsilon_0` and `r`. However the number of parameter updates `r` per batch is not updated as the paper suggested in this implementation because of number overflow issue as `r` gets very huge. Instead `r += 10` is used.

### Environment Setup
To install all libraries/dependencies used in this project, run
```bash
pip3 install -r requirement.txt
```
All TensorFlow model checkpoints can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1tWk1iaQz1mhw6bzh4mrBz4d2SrVNKGuX? usp=sharing).

### Performance Evaluation
To run a model:
```bash
python3 run.py --corpora=corpora_option --model=model_choice --train --epoch=num_epochs --batch=batch_szie --order=n
```
 - `corpora_option` is the data to use: either `brown` or `wiki`; default is `brown`
 - `model_choice`   is the MLP models to choose from: `1`, `3`, `5`, `7`, or `9`; default is `1`
 - `--train`        is the flag to train and generate new checkpoints; default is `False`
 - `num_epochs`     is the integer number of epochs used for training; default is `15`
 - `batch`          is the batch size for model training or evaluation; default is `256`
 - `n`              is the order of model; default `5`, which means a 4-word sequence followed by a 1-word prediction

The following table contains the result using Brown corpus with order of the model `n` = 5, `batch_size` = 256, and `epoch` = 15.

Note that due to initialization of truncated normal variables, the reproduced results may be slightly different.

| Brown Corpora | n | h   | m  | direct | train | valid | test |
|--------------|---|-----|----|--------|-------|-------|------|
| MLP1         | 5 | 50  | 60 | yes    | 136   | 346   | 349  |
| MLP3         | 5 | 0   | 60 | yes    | 115   | 451   | 448  |
| MLP5         | 5 | 50  | 30 | yes    | 205   | 489   | 320  |
| MLP7         | 5 | 50  | 30 | yes    | 191   | 321   | 321  |
| MLP9         | 5 | 100 | 30 | no     | 287   | 334   | 335  |

The following table contains the result using Wiki-text 2 corpus with order of the model `n` = 5, `batch_size` = 256, and `epoch` = 15.

| Wiki-text 2 Corpora | n | h   | m  | direct | train | valid | test |
|--------------|---|-----|----|--------|-------|-------|------|
| MLP7         | 5 | 50  | 30 | yes    | 178 |  156  | 138 |

One interesting thing observed is that the training perplexity can go lower than the paper suggested, but validation and test perplexities are not as good as the ones mentioned in the paper. The reason is not clear but most likely be attributed to the corpora discrepancies aforementioned and the learning rate tuning. And of course, due to faster convergence using Adam optimizer, the results tend to be overfit and thus give higher validation and test perplexities.
