import os
import re
import collections

import nltk
nltk.download('brown') # reference: http://www.nltk.org/nltk_data/
from nltk.corpus import brown

import numpy as np
import tensorflow as tf
from tqdm import tqdm

words = brown.words()
sents = brown.sents()

def map_words_sents(words, sents):
    # select vocabulary
    vocab = {}
    vocab['UNK'] = 0
    word_count_sorted = sorted(collections.Counter(words).items(), key=lambda item: item[1])
    for item in word_count_sorted:
        if item[1] > 3: # if word frequency < 3
            vocab[item[0]] = len(vocab)
        else:
            vocab['UNK'] += 1
    vocab_reversed = dict(zip(vocab.values(), vocab.keys()))

    # map word to index number
    sents_mapped = []
    for sent in sents:
        sents_mapped.append([vocab.get(word, 0) for word in sent])
    return sents_mapped, vocab, vocab_reversed

sents_mapped, vocab, vocab_reversed = map_words_sents(words, sents)

n = 5 # order of the model

def generate_data(sents_mapped, n, vocab_size, training=True):
    data = [] # sets of n-word sequence
    labels = [] # sets of 1-word prediction

    for sent in sents_mapped:
        beginning_index = 0
        end_index = n - 1
        while end_index < len(sent):
            feed = sent[beginning_index:end_index]
            target = sent[end_index]
            data.append(feed)
            arr = np.zeros((vocab_size)) # one-hot encoding
            arr[target] = 1
            labels.append(arr)
            end_index += 1
            beginning_index += 1

    return data, labels

data, labels = generate_data(sents_mapped, n, len(vocab), training=True)

def generator(data, labels, training=True):
    if training:
        i = 0
        end = int(len(data)*0.8)
    else:
        i = int(len(data)*0.8) + 1
        end = len(data) - 1

    while True:
        yield np.array(data[i]), np.array(labels[i])
        if i == end:
            i = 0
        else:
            i += 1

training_data = generator(data, labels, training=True)
training_steps = len(data[:int(len(data)*0.8)])

V = len(vocab) # vocabulary size
m = 60 # embedding size
h = 50
epsilon_0 = 10**(-3)

# total number of parameters updates (from W, U, H, d, b, and words vectors from C) per training step
t = V*(n-1)*m + V*h + h*(n-1)*m + h + V + m*(n-1)
r = 10**(-8) # decrease factor

# num_epochs = 1
num_epochs = 15

weight_decay = 10**(-4)

# model: y = b + Wx + Utanh(d + Hx)
graph = tf.Graph()
with graph.as_default(), tf.device('/device:GPU:0'):
    with tf.name_scope('inputs'):
        words = tf.placeholder(tf.int32, shape=[n-1])
        y = tf.placeholder(tf.int32, shape=[V])
        epsilon_t = tf.placeholder(tf.float64, shape=None)

    with tf.name_scope('C'):
        C = tf.Variable(tf.random_uniform([V, m], -1.0, 1.0))

    with tf.name_scope('x'):
        x = tf.reshape(tf.nn.embedding_lookup(C, words), [-1, 1])

    with tf.name_scope('H'):
        H = tf.Variable(tf.truncated_normal([h, (n-1)*m], stddev=1.0/np.sqrt((n-1)*m)))

    with tf.name_scope('U'):
        U = tf.Variable(tf.truncated_normal([V, h], stddev=1.0/np.sqrt(h)))

    with tf.name_scope('W'):
        W = tf.Variable(tf.truncated_normal([V, (n-1)*m], stddev=1.0/np.sqrt((n-1)*m)))

    with tf.name_scope('d'):
        d = tf.Variable(tf.zeros([h, 1]))

    with tf.name_scope('b'):
        b = tf.Variable(tf.zeros([V, 1]))

    with tf.name_scope('logits'):
        logits = b + tf.matmul(W, x) + tf.matmul(U, tf.tanh(d + tf.matmul(H, x)))

    prob = tf.nn.softmax(logits=logits, axis=0)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.reshape(y, [1, -1]), logits=tf.reshape(logits, [1,-1])))

    tf.summary.scalar('loss', loss)

    with tf.name_scope('optimizer'):
        Custom_Optimizer = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.GradientDescentOptimizer)
        optimizer = Custom_Optimizer(weight_decay=weight_decay, learning_rate=epsilon_t).minimize(loss)
#         optimizer = tf.train.GradientDescentOptimizer(learning_rate=epsilon_t).minimize(loss)

    # merge all summaries
    summary_merged = tf.summary.merge_all()

    # create a saver
    saver = tf.train.Saver()

print('graph built')

# create the directory for TensorBoard variables if there is not
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

num_steps = training_steps
# num_steps = 100
parameter_updates = 0

with tf.Session(graph=graph) as session:
    writer = tf.summary.FileWriter(log_dir, session.graph)

    # initialize variables
    tf.global_variables_initializer().run()
    total_loss = 0
    perplexity_exponent = 0

    for epoch in np.arange(num_epochs):
        for step in tqdm(np.arange(num_steps)):
            data_training, label = next(training_data)
            learning_rate = epsilon_0

            # collect runtime statistics
            run_metadata = tf.RunMetadata()

            _, loss_step, prob_step, summary = session.run([optimizer, loss, prob, summary_merged],
                                                            feed_dict={words:data_training, y:label, epsilon_t:learning_rate},
                                                            run_metadata=run_metadata)
            learning_rate = epsilon_0/(1+r*t)
            parameter_updates += t
            total_loss += loss_step
    #         perplexity_exponent += np.log(prob_step[np.argmax(label)][0])

            # record summaries
            writer.add_summary(summary, step)
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step {}'.format(step))

            if step % 2000 == 0 and step > 0:
                print('average loss at step ', step, ':', loss_step)
    #             print('perplexity at step', step, ':', np.exp(-perplexity_exponent/step))

    # save the model
    saver.save(session, os.path.join(log_dir, 'model.ckpt'))
    writer.close()
