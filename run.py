import os
import re
import sys
import time
import pickle
import argparse
import collections

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
# nltk.download('brown') # reference: http://www.nltk.org/nltk_data/
import numpy as np
import tensorflow as tf
from tqdm import tqdm


n = 5 # order of the model
batch_size = 256
num_epochs = 15


def cleanse_corpora(file_path):
    corpora = ''
    with open(file_path, 'r') as file:
        for row in file:
            if row == '\n':
                corpora += ' '
            else:
                corpora += row.replace('\n', '')
    return corpora


def split(sents):
    sents_split = []
    words = []
    for sent in sents:
        # decompose sentence to words and punctuations
        sent_cleansed = re.findall(r"[\w'-]+|[^\w ]+", sent)
        sents_split.append(sent_cleansed)
        for word in sent_cleansed:
            words.append(word)
    return words, sents_split


def select_vocab(words):
    vocab = {}
    vocab['UNK'] = 0
    word_count_sorted = sorted(collections.Counter(words).items(), key=lambda item: item[1])
    for item in word_count_sorted:
        if item[1] > 3: # if word frequency < 3
            vocab[item[0]] = len(vocab)
        else:
            vocab['UNK'] += 1
    vocab_reversed = dict(zip(vocab.values(), vocab.keys()))
    return vocab, vocab_reversed


def tokenize_sents(vocab, sents_split):
    '''map words in a sentence to index numbers'''
    sents_mapped = []
    for sent in sents_split:
        sents_mapped.append([vocab.get(word, 0) for word in sent])
    return np.array(sents_mapped)


def generate_data(sents_mapped):
    data = [] # sets of n-word sequence
    labels = [] # sets of 1-word prediction

    for sent in sents_mapped:
        beginning_index = 0
        end_index = n - 1
        sent_len = len(sent)

        # skip sentence whose length is smaller than order of the model
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


def num_batches(data):
    return len(data) // batch_size + 1


def preprocess_data_brown(file_path):
    print('preprocessing Brown corpora...')
    try:
        with open('corpora/brown_data.pickle', 'rb') as file:
            brown_data_dict = pickle.load(file)
        print('saved data loaded')
    except:
        start_time = time.time()
        sents = sent_tokenize(cleanse_corpora(file_path))
        # use data from NLTK directly
        # sents = nltk.corpus.brown.sents()

        training_words_count = 0
        validation_words_count = 0
        test_words_count = 0

        sents_split_training = []
        sents_split_validation = []
        sents_split_test = []

        words_training = []
        words_validation = []
        words_test = []

        for sent in sents:
            sent_cleansed = re.findall(r"[\w'-]+|[^\w ]+", sent)
            if training_words_count < 800000:
                sents_split_training.append(sent_cleansed)
                training_words_count += len(sent_cleansed)
                for word in sent_cleansed:
                    words_training.append(word)

            elif validation_words_count < 200000:
                sents_split_validation.append(sent_cleansed)
                validation_words_count += len(sent_cleansed)
                for word in sent_cleansed:
                    words_validation.append(word)

            else:
                sents_split_test.append(sent_cleansed)
                test_words_count += len(sent_cleansed)
                for word in sent_cleansed:
                    words_test.append(word)

        # select vocab from training data
        vocab, vocab_reversed = select_vocab(words_training)

        data_training, labels_training = generate_data(tokenize_sents(vocab, sents_split_training))
        data_validation, labels_validation = generate_data(tokenize_sents(vocab, sents_split_validation))
        data_test, labels_test = generate_data(tokenize_sents(vocab, sents_split_test))
        print('\ngenerated data in {:.4} s'.format(time.time()-start_time))

        print('vocab size:', len(vocab), '\n')
        for mode in ['training ', 'validation ', 'test ']:
            # check vocab sizes
            print(mode + 'sentence num:', len(eval('sents_split_{}'.format(mode))))
            print(mode + 'words num:', len(eval('words_{}'.format(mode))))

            # check data lengths
            print(mode + 'data len:', len(eval('data_{}'.format(mode))))
            print(mode + 'label len:', len(eval('labels_{}'.format(mode))), '\n')

        # save memory
        del words_training
        del words_validation
        del words_test
        del sents_split_training
        del sents_split_validation
        del sents_split_test

        brown_data_dict = {'data': {'training': data_training, 'validation': data_validation, 'test': data_test},
                            'labels': {'training': labels_training, 'validation': labels_validation, 'test': labels_test},
                            'batches': {'training': num_batches(data_training), 'validation': num_batches(data_validation), 'test': num_batches(data_test)},
                            'vocab': vocab}

        print('brown_data_dict size: {:.3} MB\n'.format(sys.getsizeof(brown_data_dict) / 1024**2))

        # save processed data
        with open('corpora/brown_data.pickle', 'wb') as file:
            pickle.dump(brown_data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    return brown_data_dict


def preprocess_data_wiki():
    print('preprocessing Wikitext-2 corpora...')
    try:
        with open('corpora/wiki_data.pickle', 'rb') as file:
            wiki_data_dict = pickle.load(file)
        print('saved data loaded')
    except:
        start_time = time.time()
        sents_training = sent_tokenize(cleanse_corpora('corpora/wiki.train.txt'))
        words_training, sents_split_training = split(sents_training)
        # select vocab from training data
        vocab, vocab_reversed = select_vocab(words_training)
        data_training, labels_training = generate_data(tokenize_sents(vocab, sents_split_training))

        sents_validation = sent_tokenize(cleanse_corpora('corpora/wiki.valid.txt'))
        words_validation, sents_split_validation = split(sents_validation)
        data_validation, labels_validation = generate_data(tokenize_sents(vocab, sents_split_validation))

        sents_test = sent_tokenize(cleanse_corpora('corpora/wiki.test.txt'))
        words_test, sents_split_test = split(sents_test)
        data_test, labels_test = generate_data(tokenize_sents(vocab, sents_split_test))
        print('\ngenerated training data in {:.4} s\n'.format(time.time()-start_time))

        print('vocab size:', len(vocab), '\n')
        for mode in ['training ', 'validation ', 'test ']:
            # check vocab sizes
            print(mode + 'sentence num:', len(eval('sents_split_{}'.format(mode))))
            print(mode + 'words num:', len(eval('words_{}'.format(mode))))

            # check data lengths
            print(mode + 'data len:', len(eval('data_{}'.format(mode))))
            print(mode + 'label len:', len(eval('labels_{}'.format(mode))), '\n')

        # save memory
        del sents_training
        del sents_validation
        del sents_test
        del words_training
        del words_validation
        del words_test
        del sents_split_training
        del sents_split_validation
        del sents_split_test

        wiki_data_dict = {'data': {'training': data_training, 'validation': data_validation, 'test': data_test},
                            'labels': {'training': labels_training, 'validation': labels_validation, 'test': labels_test},
                            'batches': {'training': num_batches(data_training), 'validation': num_batches(data_validation), 'test': num_batches(data_test)},
                            'vocab': vocab}

        print('wiki_data_dict size: {:.3} MB\n'.format(sys.getsizeof(wiki_data_dict) / 1024**2))

        # save processed data
        with open('corpora/wiki_data.pickle', 'wb') as file:
            pickle.dump(wiki_data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    return wiki_data_dict


def generator(data, labels, vocab_size):
    i = 0
    end = len(data) - 1

    while True:
        data_batch = []
        labels_batch = []

        for _ in range(batch_size):
            label_one_hot_encoded = np.zeros((vocab_size))
            label_one_hot_encoded[labels[i]] = 1

            data_batch.append(data[i])
            labels_batch.append(label_one_hot_encoded)
            if i == end - (n-1):
                i = 0
            else:
                i += 1

        yield np.array(data_batch), np.array(labels_batch)


class Model:
    '''
    the following graph contains MLP 1, MLP 5, MLP 7, and MLP 9 loss optimizations
    MLP 1 model: y = b + Wx + Utanh(d + Hx) where MLP1 is tanh(d + Hx)
    MLP 2 model: y = b + Wx + Utanh(d2 + hid2•MLP1) where NLP2 is tanh(d2 + HMLP1)
    MLP 3 model: y = b + Wx + Utanh(d3 + hid3•MLP2) where NLP3 is tanh(d3 + HMLP2)
    MLP 4 model: y = b + Wx + Utanh(d4 + hid4•MLP3)
    MLP 5 model: y = b + Wx + Utanh(d5 + hid5•MLP4)
    MLP 6 model: y = b + Wx + Utanh(d6 + hid6•MLP5)
    MLP 7 model: y = b + Wx + Utanh(d7 + hid7•MLP6)
    MLP 8 model: y = b + Wx + Utanh(d8 + hid8•MLP7)
    MLP 9 model: y = b + Utanh(d9 + hid9•MLP8) (no direct connections)
    Note that h is nonzero in above models
    '''
    def __init__(self, name, V, batch_size=batch_size, weight_decay=10**(-4)):
        if 'MLP1' in name:
            h = 50
            m = 60
        elif 'MLP5' in name or 'MLP7' in name:
            h = 50
            m = 30
        elif 'MLP9' in name:
            h = 100
            m = 30

        self.name = name
        self.V = V
        self.h = h
        self.m = m

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.words = tf.placeholder(tf.int32, shape=[batch_size, n-1])
                self.y = tf.placeholder(tf.int32, shape=[batch_size, V])
                self.epsilon_t = tf.placeholder(tf.float64, shape=None)

            with tf.name_scope('parameters'):
                C = tf.Variable(tf.random_uniform([V, m], -1.0, 1.0))
                x = tf.transpose(tf.reshape(tf.nn.embedding_lookup(C, self.words), [-1, (n-1)*m])) # [(n-1)*m, batch_size]
                H = tf.Variable(tf.truncated_normal([h, (n-1)*m], stddev=1.0/np.sqrt((n-1)*m)))
                U = tf.Variable(tf.truncated_normal([V, h], stddev=1.0/np.sqrt(h)))
                W = tf.Variable(tf.truncated_normal([V, (n-1)*m], stddev=1.0/np.sqrt((n-1)*m)))

                b = tf.Variable(tf.zeros([V, batch_size]))
                d = tf.Variable(tf.zeros([h, batch_size]))
                d2 = tf.Variable(tf.zeros([h, batch_size]))
                d3 = tf.Variable(tf.zeros([h, batch_size]))
                d4 = tf.Variable(tf.zeros([h, batch_size]))
                d5 = tf.Variable(tf.zeros([h, batch_size]))
                d6 = tf.Variable(tf.zeros([h, batch_size]))
                d7 = tf.Variable(tf.zeros([h, batch_size]))
                d8 = tf.Variable(tf.zeros([h, batch_size]))
                hid2 = tf.Variable(tf.truncated_normal([h, h], stddev=1.0/np.sqrt(h)))
                hid3 = tf.Variable(tf.truncated_normal([h, h], stddev=1.0/np.sqrt(h)))
                hid4 = tf.Variable(tf.truncated_normal([h, h], stddev=1.0/np.sqrt(h)))
                hid5 = tf.Variable(tf.truncated_normal([h, h], stddev=1.0/np.sqrt(h)))
                hid6 = tf.Variable(tf.truncated_normal([h, h], stddev=1.0/np.sqrt(h)))
                hid7 = tf.Variable(tf.truncated_normal([h, h], stddev=1.0/np.sqrt(h)))
                hid8 = tf.Variable(tf.truncated_normal([h, h], stddev=1.0/np.sqrt(h)))

            with tf.name_scope('direct_connections'):
                Wx = tf.matmul(W, x)

            with tf.name_scope('MLPs'):
                MLP1 = d + tf.matmul(H, x)
                MLP2 = d2 + tf.matmul(hid2, tf.tanh(MLP1))
                MLP3 = d3 + tf.matmul(hid3, tf.tanh(MLP2))
                MLP4 = d4 + tf.matmul(hid4, tf.tanh(MLP3))
                MLP5 = d5 + tf.matmul(hid5, tf.tanh(MLP4))
                MLP6 = d6 + tf.matmul(hid6, tf.tanh(MLP5))
                MLP7 = d7 + tf.matmul(hid7, tf.tanh(MLP6))
                MLP8 = d8 + tf.matmul(hid8, tf.tanh(MLP7))

            with tf.name_scope('MLP_logits'):
                MLP1_logits = b + Wx + tf.matmul(U, tf.tanh(MLP1))
                MLP5_logits = b + Wx + tf.matmul(U, tf.tanh(MLP5))
                MLP7_logits = b + Wx + tf.matmul(U, tf.tanh(MLP7))
                MLP9_logits = b + tf.matmul(U, tf.tanh(MLP8)) # no direct connections

            MLP1_prob = tf.nn.softmax(logits=MLP1_logits, axis=0)
            MLP5_prob = tf.nn.softmax(logits=MLP5_logits, axis=0)
            MLP7_prob = tf.nn.softmax(logits=MLP7_logits, axis=0)
            MLP9_prob = tf.nn.softmax(logits=MLP9_logits, axis=0)

            with tf.name_scope('MLP_losses'):
                MLP1_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=tf.transpose(MLP1_logits)))
                MLP5_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=tf.transpose(MLP5_logits)))
                MLP7_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=tf.transpose(MLP7_logits)))
                MLP9_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=tf.transpose(MLP9_logits)))

            tf.summary.scalar('MLP1_loss', MLP1_loss)
            tf.summary.scalar('MLP5_loss', MLP5_loss)
            tf.summary.scalar('MLP7_loss', MLP7_loss)
            tf.summary.scalar('MLP9_loss', MLP9_loss)

            with tf.name_scope('optimizers'):
                Custom_Optimizer = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
                MLP1_optimizer = Custom_Optimizer(weight_decay=weight_decay, learning_rate=self.epsilon_t).minimize(MLP1_loss)
                MLP5_optimizer = Custom_Optimizer(weight_decay=weight_decay, learning_rate=self.epsilon_t).minimize(MLP5_loss)
                MLP7_optimizer = Custom_Optimizer(weight_decay=weight_decay, learning_rate=self.epsilon_t).minimize(MLP7_loss)
                MLP9_optimizer = Custom_Optimizer(weight_decay=weight_decay, learning_rate=self.epsilon_t).minimize(MLP9_loss)

            # merge all summaries
            summary_merged = tf.summary.merge_all()

        version = name[-1]
        self.fetches = [eval('MLP{}_optimizer'.format(version)),
                        eval('MLP{}_loss'.format(version)),
                        eval('MLP{}_prob'.format(version)),
                        summary_merged]


class MLP3:
    '''
    # the following graph contains MLP 3 with h = 0 specifically
    # MLP 3 model: y = b3 + Wx + M3(b2 + M1tanh(b1 + Wx))
    '''
    def __init__(self, name, V, batch_size=batch_size, weight_decay=10**(-4)):
        m = 60

        self.name = name
        self.V = V
        self.h = 0
        self.m = m

        self.graph = tf.Graph()

        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.words = tf.placeholder(tf.int32, shape=[batch_size, n-1])
                self.y = tf.placeholder(tf.int32, shape=[batch_size, V])
                self.epsilon_t = tf.placeholder(tf.float64, shape=None)

            with tf.name_scope('parameters'):
                C = tf.Variable(tf.random_uniform([V, m], -1.0, 1.0))
                x = tf.transpose(tf.reshape(tf.nn.embedding_lookup(C, self.words), [-1, (n-1)*m])) # [(n-1)*m, batch_size]
                W = tf.Variable(tf.truncated_normal([V, (n-1)*m], stddev=1.0/np.sqrt((n-1)*m)))
                M1 = tf.Variable(tf.truncated_normal([(n-1)*m, (n-1)*m], stddev=1.0/np.sqrt(V)))
                M2 = tf.Variable(tf.truncated_normal([(n-1)*m, (n-1)*m], stddev=1.0/np.sqrt(V)))
                M3 = tf.Variable(tf.truncated_normal([V, (n-1)*m], stddev=1.0/np.sqrt(V)))
                b1 = tf.Variable(tf.zeros([(n-1)*m, batch_size]))
                b2 = tf.Variable(tf.zeros([(n-1)*m, batch_size]))
                b3 = tf.Variable(tf.zeros([V, batch_size]))

            with tf.name_scope('direct_connections'):
                Wx = tf.matmul(W, x)

            with tf.name_scope('MLPs'):
                MLP1 = b1 + tf.matmul(M1, x)
                MLP2 = b2 + tf.matmul(M2, tf.tanh(MLP1))
                MLP3 = tf.matmul(M3, MLP2)

            with tf.name_scope('MLP_logits'):
                MLP3_logits = b3 + Wx + MLP3

            MLP3_prob = tf.nn.softmax(logits=MLP3_logits, axis=0)

            with tf.name_scope('MLP_losses'):
                MLP3_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=tf.transpose(MLP3_logits)))

            tf.summary.scalar('MLP3_loss', MLP3_loss)

            with tf.name_scope('optimizers'):
                Custom_Optimizer = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
                MLP3_optimizer = Custom_Optimizer(weight_decay=weight_decay, learning_rate=self.epsilon_t).minimize(MLP3_loss)

            # merge all summaries
            summary_merged = tf.summary.merge_all()

        self.fetches = [MLP3_optimizer, MLP3_loss, MLP3_prob, summary_merged]


def train(model, data, num_batches):
    # create TensorBoard directory
    log_dir = model.name + '_log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with tf.Session(graph=model.graph) as session:
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(log_dir, session.graph)

        # initialize variables
        tf.global_variables_initializer().run()

        epsilon_0 = 10**(-3)
        r = 10**(-8) # decrease factor
        # number of parameters updates (from W, U, H, d, b, and words vectors from C) per training step
        t = model.V*(n-1)*model.m +  model.V*model.h + model.h*(n-1)*model.m + model.h + model.V + model.m*(n-1)

        learning_rate = epsilon_0
        batches_total = 0
        loss_total = 0
        perplexity_exponent = 0
        perplexity_exponent_total = 0

        for epoch in tqdm(np.arange(num_epochs)):
            print('epoch:', epoch + 1)
            for batch in tqdm(np.arange(num_batches)):
                data_training, label = next(data)
                feed_dict={model.words:data_training, model.y:label, model.epsilon_t:learning_rate}

                # collect runtime statistics
                run_metadata = tf.RunMetadata()
                _, loss_batch, prob_batch, summary = session.run(model.fetches, feed_dict=feed_dict, run_metadata=run_metadata)

                # update learning rate
                learning_rate = epsilon_0/(1+r*t)
                t += 10

                batches_total += 1
                loss_total += loss_batch

                prob_batch = prob_batch.T # [batch_size, vocab_size]
                perplexity_exponent = np.sum(np.log(prob_batch[np.arange(len(prob_batch)), np.argmax(label, axis=1)]))
                perplexity_exponent_total += perplexity_exponent

                # record summaries
                writer.add_summary(summary, batches_total)
                if batch == (num_batches - 1):
                    writer.add_run_metadata(run_metadata, 'epoch{} batch {}'.format(epoch, batch))

                if batch % 100 == 0 and batch > 0:
                    print('loss at batch', batches_total, ':', loss_batch)
                    print('average loss per word so far: {:.3}'.format(loss_total/batches_total/batch_size))
                    print('average perplexity per word so far: {:.3}'.format(np.exp(-perplexity_exponent_total/batches_total/batch_size)))

        # save model
        saver.save(sess=session, save_path=os.path.join(log_dir, '{}.ckpt'.format(model.name)))
        writer.close()

        # record results
        if not os.path.exists('results'):
            os.makedirs('results')
        file = open('results/{}.txt'.format(model.name), 'a')
        file.write('final training perplexity: {}\n'.format(np.exp(-perplexity_exponent_total/batches_total/batch_size)))
        file.close()


def evaluate(model, evaluation_data, num_batches, validation_flag=1):
    with tf.Session(graph=model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=os.path.join(model.name + '_log', '{}.ckpt'.format(model.name)))

        batches_total = 0
        loss_total = 0
        perplexity_exponent = 0
        perplexity_exponent_total = 0

        for batch in tqdm(np.arange(num_batches)):
            data_evaluating, label = next(evaluation_data)
            feed_dict={model.words:data_evaluating, model.y:label}

            loss_batch, prob_batch = sess.run([model.fetches[1], model.fetches[2]], feed_dict=feed_dict)

            batches_total += 1
            loss_total += loss_batch

            prob_batch = prob_batch.T # [batch_size, vocab_size]
            perplexity_exponent = np.sum(np.log(prob_batch[np.arange(len(prob_batch)), np.argmax(label, axis=1)]))
            perplexity_exponent_total += perplexity_exponent

            if batch % 100 == 0 and batch > 0:
                print('loss at batch', batches_total, ':', loss_batch)
                print('average loss per word so far: {:.3}'.format(loss_total/batches_total/batch_size))
                print('average perplexity per word so far: {:.3}'.format(np.exp(-perplexity_exponent_total/batches_total/batch_size)))

        if validation_flag:
            suffix = 'validation'
        else:
            suffix = 'test'
        if not os.path.exists('results'):
            os.makedirs('results')
        file = open('results/{}.txt'.format(model.name), 'a')
        file.write('final {} perplexity: {}\n'.format(suffix, np.exp(-perplexity_exponent_total/batches_total/batch_size)))
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='command line options; default to evaluate MLP1 on Brown corpus')
    parser.add_argument('--corpora', action="store", dest="corpora_option", default='brown', help="data to use: either 'brown' or 'wiki'")
    parser.add_argument('--model', action="store", dest="model_choice", default='1', help="MLP models to choose: '1', '3', '5', '7', or '9'")
    parser.add_argument('--train', action="store_true", dest="training", default=False, help='flag to train selected model and generate new checkpoints')
    parser.add_argument('--epoch', action="store", dest="num_epochs", default=15, type=int, help='integer number of epochs for training')
    parser.add_argument('--batch', action="store", dest="batch_size", default=256, type=int, help='integer number of batch size for model training or evaluation')
    parser.add_argument('--order', action="store", dest="n", default=5, type=int, help='order of the model')
    inputs = parser.parse_args()

    corpora_option = inputs.corpora_option
    model_choice = inputs.model_choice
    num_epochs = inputs.num_epochs
    batch_size = inputs.batch_size
    n = inputs.n

    # use Brown corpora
    if corpora_option == 'brown':
        corpora_name = 'Brown'
        data_dict = preprocess_data_brown('corpora/brown.txt')

    # use Wiki corpora
    if corpora_option == 'wiki':
        corpora_name = 'Wiki'
        data_dict = preprocess_data_wiki()

    vocab_len = len(data_dict['vocab'])

    data_training = generator(data_dict['data']['training'], data_dict['labels']['training'], vocab_len)
    data_validation = generator(data_dict['data']['validation'], data_dict['labels']['validation'], vocab_len)
    data_test = generator(data_dict['data']['test'], data_dict['labels']['test'], vocab_len)

    num_batches_training = data_dict['batches']['training']
    num_batches_validation = data_dict['batches']['validation']
    num_batches_test = data_dict['batches']['test']

    if model_choice == '1':
        model = Model(name='{}_MLP1'.format(corpora_name), V=vocab_len)
    elif model_choice == '3':
        model = MLP3(name='{}_MLP3'.format(corpora_name), V=vocab_len)
    elif model_choice == '5':
        model = Model(name='{}_MLP5'.format(corpora_name), V=vocab_len)
    elif model_choice == '7':
        model = Model(name='{}_MLP7'.format(corpora_name), V=vocab_len)
    elif model_choice == '9':
        model = Model(name='{}_MLP9'.format(corpora_name), V=vocab_len)

    print('\nrunning MLP{} model on {} corpora...'.format(model_choice, corpora_name))
    print('order of model:', n)
    print('num_epochs:', num_epochs)
    print('batch size:', batch_size, '\n')

    # train, validate, and test
    if inputs.training:
        train(model, data_training, num_batches_training)
    evaluate(model, data_validation, num_batches_validation, validation_flag=1)
    evaluate(model, data_test, num_batches_test, validation_flag=0)
