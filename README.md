# Statistical-Language-Modeling

## Deep Learning for Natural Language Processing
This repository contains a modern implementation of the classic paper [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) by [Yoshua Bengio](https://en.wikipedia.org/wiki/Yoshua_Bengio) et al. in 2003 using [TensorFlow](https://www.tensorflow.org/). All results were obtained with an NVIDIA's [GeForce RTX 2080 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2080-ti/) Graphics Card.

This implementation uses the same setting as the one mentioned in the paper: the first 800,000 words for training, the following 200,000 words for validation, and the remaining words for testing. However, because the [Brown corpora](https://en.wikipedia.org/wiki/Brown_Corpus) used in this project has minor discrepancy in word numbers as compared to the Brown corpora used in the paper, the final vocabulary size of the word embedding use is slightly different from the one mentioned in the paper. Besides Brown corpora, [Wikitext-2 corpora](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) is used to evaluate MLP7's performance in particular.

In addition, the weight decay used in this implementation leverages TensorFlow's built-in function `tf.contrib.opt.extend_with_decoupled_weight_decay` which includes biases, which are actually not included in the paper. But for convenience of implementing parameters' weight decay, this function is used anyway.

Another major difference worth to note is that Adam optimizer instead of stochastic gradient descent optimizer is used extensively in this project for faster convergence.

As for the gradually decreasing learning rate, this implementation uses the given `epsilon_0` and `r`. However the number of parameter updates `r` per batch is not updated as the paper suggested in this implementation because of number overflow issue as `r` gets very huge. Instead `r += 10` is used.

### Environment Setup
To install all libraries/dependencies used in this project, run
```bash
pip3 install -r requirement.txt
```
Download all TensorFlow model checkpoints from [Google Drive](https://drive.google.com/drive/folders/1tWk1iaQz1mhw6bzh4mrBz4d2SrVNKGuX?usp=sharing) to the same directory of `run.py`.

### Performance Evaluation
To train a model, run:
```bash
python3 run.py --corpora=corpora_option --model=model_choice --train --epoch=num_epochs --batch=batch_size --order=n
```

To evaluate a model from saved checkpoints, run:
```bash
python3 run.py --corpora=corpora_option --model=model_choice
```

- `corpora_option` is the data to use: either `brown` or `wiki`; default is `brown`
- `model_choice`   is the MLP models to choose from: `1`, `3`, `5`, `7`, or `9`; default is `1`
- `--train`        is the flag to train and generate new checkpoints; default is `False`
- `num_epochs`     is the integer number of epochs used for training; default is `15`
- `batch`          is the batch size for model training or evaluation; default is `256`
- `n`              is the order of model; default `5`, which means a 4-word sequence followed by a 1-word prediction

To visualize a model's computation graph via TensorBoard, run:
 ```bash
tensorboard --logdir=model_checkpoint_dir
 ```
then copy the generated link to a browser.

Note that all perplexities calculated for each model will be saved in `results` directory generated on the fly.

The following two table contain the results (saved also at [Google Drive](https://drive.google.com/drive/folders/1tWk1iaQz1mhw6bzh4mrBz4d2SrVNKGuX?usp=sharing)) using Brown corpora and Wiki-text 2 corpora with order of the model `n` = 5, `batch_size` = 256, and `epoch` = 15. `h` is the number of hidden units, m is the number of word features for MLPs, and direct indicates whether there are direct connections from word features to outputs. More details can be found in the paper. Note that due to initialization of truncated normal variables in word embeddings and other weight matrices, the reproduced results may be slightly different.

| Brown Corpora | n | h   | m  | direct | train | valid | test |
|--------------|---|-----|----|--------|-------|-------|------|
| MLP1         | 5 | 50  | 60 | yes    | 122   | 264   | 277  |
| MLP3         | 5 | 0   | 60 | yes    | 105   | 311   | 322  |
| MLP5         | 5 | 50  | 30 | yes    | 173   | 268   | 280  |
| MLP7         | 5 | 50  | 30 | yes    | 174   | 261   | 273  |
| MLP9         | 5 | 100 | 30 | no     | 261   | 302   | 328  |

| Wiki-text 2 Corpora | n | h   | m  | direct | train | valid | test |
|--------------|---|-----|----|--------|-------|-------|------|
| MLP7         | 5 | 50  | 30 | yes    | 185 |  154  | 137 |

In general, the training and validation perplexities are lower than the paper presented and they can go even lower with more training, but test perplexities are a little bit higher than the ones mentioned in the paper sometimes. This suggests more epochs of training may leads to better results. One interesting thing is that our MLP9 doesn't overfit very much like the paper's version did and achieves OK validation and test perplexities (more epochs may lead to better result). In addition, due to faster convergence using Adam optimizer, the models tend to overfit on training data and thus give higher validation or test perplexities.
