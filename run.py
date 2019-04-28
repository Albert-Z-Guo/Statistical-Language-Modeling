import os
import re
import sys
import time
import pickle
import collections

import nltk
nltk.download('brown') # reference: http://www.nltk.org/nltk_data/
import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook
from tqdm import tqdm


def map_words_sents(words, sents):
    # select vocabulary, all punctuations included
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
    return np.array(sents_mapped), vocab, vocab_reversed


# downlaod corpus
words = nltk.corpus.brown.words()
sents = nltk.corpus.brown.sents()
sents_mapped, vocab, vocab_reversed = map_words_sents(words, sents)

print('sentence num:', len(sents))
print('words num:', len(words))
print('vocab size:', len(vocab))

# reduce memory
del words
del sents


n = 5 # order of the model
def generate_data(sents_mapped, n):
    data = [] # sets of n-word sequence
    labels = [] # sets of 1-word prediction

    for sent in sents_mapped:
        beginning_index = 0
        end_index = n - 1
        sent_len = len(sent)

        # skip too short sentence
        if sent_len < n:
            continue

        while end_index < sent_len:
            feed = sent[beginning_index:end_index]
            target = sent[end_index]
            data.append(feed)
            labels.append(target)
            end_index += 1
            beginning_index += 1

    return np.array(data), np.array(labels)


'''try reading saved data, if no data found, generate data'''
try:
    with open('data.pickle', 'rb') as file:
        data = pickle.load(file)

    with open('labels.pickle', 'rb') as file:
        labels = pickle.load(file)
except:


    # check variable sizes
    print('sents_mapped size: {:.3} MB'.format(sys.getsizeof(sents_mapped) / 1024**2))
    print('vocab size: {:.3} MB'.format(sys.getsizeof(vocab) / 1024**2))
    print('vocab_reversed size: {:.3} MB'.format(sys.getsizeof(vocab_reversed) / 1024**2))

    start_time = time.time()
    data, labels = generate_data(sents_mapped, n)
    print('\ngenerated data in {:.4} s'.format(time.time()-start_time))

    print('\ndata len:', len(data))
    print('label len:', len(labels))

    # check variable sizes
    print('data size: {:.3} MB'.format(sys.getsizeof(data) / 1024**2))
    print('labels size: {:.3} MB'.format(sys.getsizeof(labels) / 1024**2))

    # save generated data
    with open('data.pickle', 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('labels.pickle', 'wb') as file:
        pickle.dump(labels, file, protocol=pickle.HIGHEST_PROTOCOL)

    # reduce memory
    del sents_mapped


batch_size = 64

def generator(data, labels, vocab_size, training=True):
    if training:
        i = 0
        end = int(len(data)*0.8)
    else:
        i = int(len(data)*0.8) + 1
        end = len(data) - 1

    while True:
        data_batch = []
        labels_batch = []

        for _ in range(batch_size):
            label_one_hot_encoded = np.zeros((vocab_size))
            label_one_hot_encoded[labels[i]] = 1

            data_batch.append(data[i])
            labels_batch.append(label_one_hot_encoded)
            if i == end:
                i = 0
            else:
                i += 1

        yield np.array(data_batch), np.array(labels_batch)


training_steps = len(data[:int(len(data)*0.8)])//128 + 1
training_data = generator(data, labels, len(vocab), training=True)


# the following graph contains MLP 1, MLP 5, MLP 7, and MLP 9 loss optimizations
# MLP 1 model: y = b + Wx + Utanh(d + Hx) where MLP 1 is d + Hx
h = 0
V = len(vocab) # vocabulary size
m = 60 # embedding size
weight_decay = 10**(-4)

# graph = tf.Graph()
# with graph.as_default():
#     with tf.name_scope('inputs'):
#         words = tf.placeholder(tf.int32, shape=[batch_size, n-1])
#         y = tf.placeholder(tf.int32, shape=[batch_size, V])
#         epsilon_t = tf.placeholder(tf.float64, shape=None)
#
#     with tf.name_scope('parameters'):
#         C = tf.Variable(tf.random_uniform([V, m], -1.0, 1.0))
#         x = tf.transpose(tf.reshape(tf.nn.embedding_lookup(C, words), [-1, (n-1)*m])) # [(n-1)*m, batch_size]
#         H = tf.Variable(tf.truncated_normal([h, (n-1)*m], stddev=1.0/np.sqrt((n-1)*m)))
#         U = tf.Variable(tf.truncated_normal([V, h], stddev=1.0/np.sqrt(h)))
#         W = tf.Variable(tf.truncated_normal([V, (n-1)*m], stddev=1.0/np.sqrt((n-1)*m)))
#
#         b = tf.Variable(tf.zeros([V, batch_size]))
#         d = tf.Variable(tf.zeros([h, batch_size]))
#         d2 = tf.Variable(tf.zeros([h, batch_size]))
#         d3 = tf.Variable(tf.zeros([h, batch_size]))
#         d4 = tf.Variable(tf.zeros([h, batch_size]))
#         d5 = tf.Variable(tf.zeros([h, batch_size]))
#         d6 = tf.Variable(tf.zeros([h, batch_size]))
#         d7 = tf.Variable(tf.zeros([h, batch_size]))
#         d8 = tf.Variable(tf.zeros([h, batch_size]))
#         hid2 = tf.Variable(tf.truncated_normal([h, h], stddev=1.0/np.sqrt(h)))
#         hid3 = tf.Variable(tf.truncated_normal([h, h], stddev=1.0/np.sqrt(h)))
#         hid4 = tf.Variable(tf.truncated_normal([h, h], stddev=1.0/np.sqrt(h)))
#         hid5 = tf.Variable(tf.truncated_normal([h, h], stddev=1.0/np.sqrt(h)))
#         hid6 = tf.Variable(tf.truncated_normal([h, h], stddev=1.0/np.sqrt(h)))
#         hid7 = tf.Variable(tf.truncated_normal([h, h], stddev=1.0/np.sqrt(h)))
#         hid8 = tf.Variable(tf.truncated_normal([h, h], stddev=1.0/np.sqrt(h)))
#
#     with tf.name_scope('direct_connections'):
#         Wx = tf.matmul(W, x)
#
#     with tf.name_scope('MLPs'):
#         MLP1 = d + tf.matmul(H, x)
#         MLP2 = d2 + tf.matmul(hid2, tf.tanh(MLP1))
#         MLP3 = d3 + tf.matmul(hid3, tf.tanh(MLP2))
#         MLP4 = d4 + tf.matmul(hid4, tf.tanh(MLP3))
#         MLP5 = d5 + tf.matmul(hid5, tf.tanh(MLP4))
#         MLP6 = d6 + tf.matmul(hid6, tf.tanh(MLP5))
#         MLP7 = d7 + tf.matmul(hid7, tf.tanh(MLP6))
#         MLP8 = d8 + tf.matmul(hid8, tf.tanh(MLP7))
#
#     with tf.name_scope('MLP_logits'):
#         MLP1_logits = b + Wx + tf.matmul(U, tf.tanh(MLP1))
#         MLP5_logits = b + Wx + tf.matmul(U, tf.tanh(MLP5))
#         MLP7_logits = b + Wx + tf.matmul(U, tf.tanh(MLP7))
#         MLP9_logits = b + tf.matmul(U, tf.tanh(MLP8)) # no direct connections
#
#     MLP1_prob = tf.nn.softmax(logits=MLP1_logits, axis=0)
#     MLP5_prob = tf.nn.softmax(logits=MLP5_logits, axis=0)
#     MLP7_prob = tf.nn.softmax(logits=MLP7_logits, axis=0)
#     MLP9_prob = tf.nn.softmax(logits=MLP9_logits, axis=0)
#
#     with tf.name_scope('MLP_losses'):
#         MLP1_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=tf.transpose(MLP1_logits)))
#         MLP5_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=tf.transpose(MLP5_logits)))
#         MLP7_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=tf.transpose(MLP7_logits)))
#         MLP9_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=tf.transpose(MLP9_logits)))
#
#     tf.summary.scalar('MLP1_loss', MLP1_loss)
#     tf.summary.scalar('MLP5_loss', MLP5_loss)
#     tf.summary.scalar('MLP7_loss', MLP7_loss)
#     tf.summary.scalar('MLP9_loss', MLP9_loss)
#
#     with tf.name_scope('optimizers'):
#         Custom_Optimizer = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
#         MLP1_optimizer = Custom_Optimizer(weight_decay=weight_decay, learning_rate=epsilon_t).minimize(MLP1_loss)
#         MLP5_optimizer = Custom_Optimizer(weight_decay=weight_decay, learning_rate=epsilon_t).minimize(MLP5_loss)
#         MLP7_optimizer = Custom_Optimizer(weight_decay=weight_decay, learning_rate=epsilon_t).minimize(MLP7_loss)
#         MLP9_optimizer = Custom_Optimizer(weight_decay=weight_decay, learning_rate=epsilon_t).minimize(MLP9_loss)
#     #         optimizer = tf.train.AdamOptimizer(learning_rate=epsilon_t).minimize(loss) # without weight decay
#
#     # merge all summaries
#     summary_merged = tf.summary.merge_all()


MLP3_graph = tf.Graph()
with MLP3_graph.as_default(), tf.device('/cpu:0'):
    with tf.name_scope('inputs'):
        words = tf.placeholder(tf.int32, shape=[batch_size, n-1])
        y = tf.placeholder(tf.int32, shape=[batch_size, V])
        epsilon_t = tf.placeholder(tf.float64, shape=None)

    with tf.name_scope('parameters'):
        C = tf.Variable(tf.random_uniform([V, m], -1.0, 1.0))
        x = tf.transpose(tf.reshape(tf.nn.embedding_lookup(C, words), [-1, (n-1)*m])) # [(n-1)*m, batch_size]
        W = tf.Variable(tf.truncated_normal([V, (n-1)*m], stddev=1.0/np.sqrt((n-1)*m)))
        M1 = tf.Variable(tf.truncated_normal([V, V], stddev=1.0/np.sqrt(V)))
        M2 = tf.Variable(tf.truncated_normal([V, V], stddev=1.0/np.sqrt(V)))
        b1 = tf.Variable(tf.zeros([V, batch_size]))
        b2 = tf.Variable(tf.zeros([V, batch_size]))
        b3 = tf.Variable(tf.zeros([V, batch_size]))

    with tf.name_scope('direct_connections'):
        Wx = tf.matmul(W, x)

    with tf.name_scope('MLPs'):
        MLP1 = b1 + tf.matmul(W, x)
        MLP2 = b2 + tf.matmul(M1, MLP1)
        MLP3 = tf.matmul(M2, MLP2)

    with tf.name_scope('MLP_logits'):
        MLP3_logits = b3 + Wx + MLP3

    MLP3_prob = tf.nn.softmax(logits=MLP3_logits, axis=0)

    with tf.name_scope('MLP_losses'):
        MLP3_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=tf.transpose(MLP3_logits)))

    tf.summary.scalar('MLP3_loss', MLP3_loss)

    with tf.name_scope('optimizers'):
        # Custom_Optimizer = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
        # MLP3_optimizer = Custom_Optimizer(weight_decay=weight_decay, learning_rate=epsilon_t).minimize(MLP3_loss)
        MLP3_optimizer = tf.train.AdamOptimizer(learning_rate=epsilon_t).minimize(MLP3_loss) # without weight decay

    # merge all summaries
    summary_merged = tf.summary.merge_all()


## create the directory for TensorBoard variables if there is not
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


epsilon_0 = 10**(-3)
r = 10**(-8) # decrease factor
# total number of parameters updates (from W, U, H, d, b, and words vectors from C) per training step
t = V*(n-1)*m + V*h + h*(n-1)*m + h + V + m*(n-1)

num_epochs = 20
num_steps = training_steps
parameter_updates = 0

# with tf.Session(graph=model.graph) as session:
with tf.Session(graph=MLP3_graph) as session:
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(log_dir, session.graph)

    # initialize variables
    tf.global_variables_initializer().run()
    total_loss = 0
    perplexity_exponent = 0

    learning_rate = epsilon_0
    total_steps = 0

    for epoch in np.arange(num_epochs):
        print('epoch:', epoch + 1)
        for step in tqdm_notebook(np.arange(num_steps)):
            data_training, label = next(training_data)

            # collect runtime statistics
            run_metadata = tf.RunMetadata()

            # _, loss_step, prob_step, summary = session.run([MLP1_optimizer, MLP1_loss, MLP1_prob, summary_merged],
            #                                                 feed_dict={words:data_training, y:label, epsilon_t:learning_rate},
            #                                                 run_metadata=run_metadata)
            #
            # _, loss_step, prob_step, summary = session.run([MLP5_optimizer, MLP5_loss, MLP5_prob, summary_merged],
            #                                                 feed_dict={words:data_training, y:label, epsilon_t:learning_rate},
            #                                                 run_metadata=run_metadata)

            # _, loss_step, prob_step, summary = session.run([MLP7_optimizer, MLP7_loss, MLP7_prob, summary_merged],
            #                                                 feed_dict={words:data_training, y:label, epsilon_t:learning_rate},
            #                                                 run_metadata=run_metadata)

            # _, loss_step, prob_step, summary = session.run([MLP9_optimizer, MLP9_loss, MLP9_prob, summary_merged],
            #                                                 feed_dict={words:data_training, y:label, epsilon_t:learning_rate},
            #                                                 run_metadata=run_metadata)

            _, loss_step, prob_step, summary = session.run([MLP3_optimizer, MLP3_loss, MLP3_prob, summary_merged],
                                                            feed_dict={words:data_training, y:label, epsilon_t:learning_rate},
                                                            run_metadata=run_metadata)

            total_steps += 1
            learning_rate = epsilon_0/(1+r*t)
            parameter_updates += t
            total_loss += loss_step
            perplexity_exponent += np.log(prob_step[np.argmax(label)][0])

            # record summaries
            writer.add_summary(summary, total_steps)
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'epoch{} step {}'.format(epoch, step))

            if step % 100 == 0 and step > 0:
                print('average loss at step ', total_steps, ':', total_loss/total_steps)
                print('perplexity at step', total_steps, ':', np.exp(-perplexity_exponent/total_steps))

    # save the model
    saver.save(session, os.path.join(log_dir, 'MLP3.ckpt'))
    writer.close()

# record results
file = open('MLP3_results.txt', 'w')
file.write('final perplexity: ' + str(np.exp(-perplexity_exponent/total_steps)))
file.close()
