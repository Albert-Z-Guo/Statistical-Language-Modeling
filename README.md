# Statistical-Language-Modeling

## Deep Learning for Natural Language Processing
This repo contains modern implementation of the classic paper [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) by [Yoshua Bengio](https://en.wikipedia.org/wiki/Yoshua_Bengio) et al. in 2003 using [Tensorflow](https://www.tensorflow.org/). All results were obtained with an NVIDIA's GeForce RTX 2080 Ti Graphics Card.

Because the Brown corpora used has minor discrepancy in word numbers as compared to the Brown corpora used in the paper, the final vocabulary size of the word embedding use is slightly different from the one mentioned in the paper. This implementation also used 80% of data for training, the following 10% for validation, and the rest 10% for testing, instead of the hard number division mentioned in the paper.

In addition, the weight decay used in this implementation leverages Tensorflow's built-in function `tf.contrib.opt.extend_with_decoupled_weight_decay` which includes biases, which are actually not included in the paper.

As for the gradually decreasing learning rate, this implementation uses the given `epsilon_0` and `r`. However the number of parameter updates `r` per batch is not updated in this implementation because of number overflow issue as `r` gets very huge.

### Environment Setup
To install all libraries/dependencies used in this project, run
```bash
pip3 install -r requirement.txt
```

### Performance Evaluation
To train a model:
```bash
python3 run.py --corpora CORPORA_OPTION --model MODEL_CHOICE --train --epoch NUM_EPOCHS
```
where  
 - `CORPORA_OPTION` is the data to use: either 'brown' or 'wiki'; default is 'brown'
 - `MODEL_CHOICE` is the MLP models to choose from: '1', '3', '5', '7', or '9'; default is '1'
 - `--train` is the flag to train and generate new checkpoints; default is False
 - `NUM_EPOCHS` is the integer number of epochs used for training; default is 15

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
| MLP7         | 5 | 50  | 30 | yes    | 185 |  155  | 137 |
