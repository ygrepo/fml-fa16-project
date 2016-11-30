from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from itertools import compress

data_index = 0

# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

def generate_batch_skipgram(data, batch_size, num_skips, skip_window):
    '''
    Batch generator for skip-gram model.

    Parameters
    ----------
    data: list of index of words
    batch_size: number of words in each mini-batch
    num_skips: number of surrounding words on both direction (2: one word ahead and one word following)
    skip_window: number of words at both ends of a sentence to skip (1: skip the first and last word of a sentence)
    '''
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

def generate_batch_cbow(data, batch_size, num_skips, skip_window):
    '''
    Batch generator for CBOW (Continuous Bag of Words).
    batch should be a shape of (batch_size, num_skips)

    Parameters
    ----------
    data: list of index of words
    batch_size: number of words in each mini-batch
    num_skips: number of surrounding words on both direction (2: one word ahead and one word following)
    skip_window: number of words at both ends of a sentence to skip (1: skip the first and last word of a sentence)
    '''
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # used for collecting data[data_index] in the sliding window
    # collect the first window of words
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # move the sliding window
    for i in range(batch_size):
        mask = [1] * span
        mask[skip_window] = 0
        batch[i, :] = list(compress(buffer, mask))  # all surrounding words
        labels[i, 0] = buffer[skip_window]  # the word at the center
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

def plot_with_labels(low_dim_embs, labels, filename='tsne_senses.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    # senses_names = load_senses_names_dict()
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        # sense_name = get_name(label, senses_names)
        plt.annotate(label,
                     #plt.annotate(sense_name,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)

def plot(w2vc, filename):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = w2vc.vocabulary_size
    low_dim_embs = tsne.fit_transform(w2vc.final_embeddings[:plot_only,:])
    labels = [w2vc.reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, filename)


class Syn2Vec(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size=128, num_skips=2, skip_window=1,
                 architecture='skip-gram', embedding_size=128, vocabulary_size=50000,
                 loss_type='nce_loss', n_neg_samples=64,
                 optimize='SGD',
                 learning_rate=1.0, n_steps=100001,
                 valid_size=16, valid_window=100, save_path='models/test_model'):
        # bind params to class
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.architecture = architecture
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.loss_type = loss_type
        self.n_neg_samples = n_neg_samples
        self.optimize = optimize
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.valid_size = valid_size
        self.valid_window = valid_window
        # # pick a list of words as validataion set
        self.pick_valid_samples()
        # # choose a batch_generator function for feed_dict
        self.choose_batch_generator()
        self.save_path = save_path

    def build_dictionaries(self, words):
        '''
        Process tokens and build dictionaries mapping between tokens and
        their indices. Also generate token count and bind these to self.
        '''

        data, count, dictionary, reverse_dictionary = build_dataset(words, self.vocabulary_size)
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        self.vocabulary_size = len(reverse_dictionary)
        if self.n_neg_samples > self.vocabulary_size:
            self.n_neg_samples = self.vocabulary_size / 2
        self.count = count
        print('Most common words (+UNK)', count[:5])
        print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
        return data

    def choose_batch_generator(self):
        if self.architecture == 'skip-gram':
            self.generate_batch = generate_batch_skipgram
        elif self.architecture == 'cbow':
            self.generate_batch = generate_batch_cbow

    def pick_valid_samples(self):
        #valid_examples = np.array(random.sample(range(self.valid_window), self.valid_size))
        valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        self.valid_examples = valid_examples

    def init_graph(self):
        '''
        Init a tensorflow Graph containing:
        input data, variables, model, loss function, optimizer
        '''
        self.graph = tf.Graph()

        with self.graph.as_default():

            # Input data.
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                #embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

                # Look up embeddings for inputs.
                if self.architecture == 'skip-gram':
                    embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)
                elif self.architecture == 'cbow':
                    embed = tf.zeros([self.batch_size, self.embedding_size])
                    for j in range(self.num_skips):
                        embed += tf.nn.embedding_lookup(embeddings, self.train_inputs[:, j])

                # Construct the variables for the NCE loss
                weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                            stddev=1.0 / math.sqrt(self.embedding_size)))
                biases = tf.Variable(tf.zeros([self.vocabulary_size]))

                # Model.

                # Compute the loss, using a sample of the negative labels each time.
                if self.loss_type == 'sampled_softmax_loss':
                    loss = tf.nn.sampled_softmax_loss(weights, biases, embed,
                                                      self.train_labels, self.n_neg_samples, self.vocabulary_size)
                elif self.loss_type == 'nce_loss':
                    # Compute the average NCE loss for the batch.
                    # tf.nce_loss automatically draws a new sample of the negative labels each
                    # time we evaluate the loss.
                    loss = tf.nn.nce_loss(weights, biases, embed,
                                          self.train_labels, self.n_neg_samples, self.vocabulary_size)
                self.loss = tf.reduce_mean(loss)

                # # Construct the SGD optimizer using a learning rate of 1.0.
                # self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)
                # Optimizer.
                if self.optimize == 'Adagrad':
                    self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
                elif self.optimize == 'Adam':
                    self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                elif self.optimize == 'SGD':
                    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

                # Compute the cosine similarity between minibatch examples and all embeddings.
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                self.normalized_embeddings = embeddings / norm
                #valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, valid_dataset)
                #self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)
                #self.similarity = tf.matmul(self.valid_embeddings, tf.transpose(self.normalized_embeddings))

                # Add variable initializer.
                self.init_op = tf.initialize_all_variables()

                # create a saver
                self.saver = tf.train.Saver()

    def fit(self, words):
        '''
        words: a list of words.
        '''
        # pre-process words to generate indices and dictionaries
        data = self.build_dictionaries(words)
        # init all variables in a tensorflow graph
        self.init_graph()
        # create a session
        self.sess = tf.Session(graph=self.graph)
        with self.sess as session:
            # We must initialize all variables before we use them.
            self.init_op.run()
            print("Initialized")

            average_loss = 0
            for step in xrange(self.n_steps):
                batch_inputs, batch_labels = self.generate_batch(data, self.batch_size, self.num_skips, self.skip_window)
                feed_dict = {self.train_inputs : batch_inputs, self.train_labels : batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0
                # # Note that this is expensive (~20% slowdown if computed every 500 steps)
                # if step % 10000 == 0:
                #     sim = self.similarity.eval()
                #     for i in xrange(valid_size):
                #         valid_word = self.reverse_dictionary[self.valid_examples[i]]
                #         top_k = 8 # number of nearest neighbors
                #         nearest = (-sim[i, :]).argsort()[1:top_k+1]
                #         log_str = "Nearest to %s:" % valid_word
                #         for k in xrange(top_k):
                #             close_word = self.reverse_dictionary[nearest[k]]
                #             log_str = "%s %s," % (log_str, close_word)
                #         print(log_str)
            final_embeddings = session.run(self.normalized_embeddings)
            self.final_embeddings = final_embeddings
            self.save(self.save_path)

    def save(self, path):
        '''
        To save trained model and its params.
        '''
        save_path = self.saver.save(self.sess,
                                    os.path.join(path, 'model.ckpt'))
        # save parameters of the model
        params = self.get_params()
        json.dump(params,
                  open(os.path.join(path, 'model_params.json'), 'wb'))

        # save dictionary, reverse_dictionary
        json.dump(self.dictionary,
                  open(os.path.join(path, 'model_dict.json'), 'wb'))
        json.dump(self.reverse_dictionary,
                  open(os.path.join(path, 'model_rdict.json'), 'wb'))

        print("Model saved in file: %s" % save_path)
        return save_path

    def restore_from_saver(self, path):
        with self.graph.as_default():
            self.saver.restore(self.sess, path)

    @classmethod
    def restore(cls, path):
        '''
        To restore a saved model.
        '''
        # load params of the model
        path_dir = os.path.dirname(path)
        params = json.load(open(os.path.join(path_dir, 'model_params.json'), 'rb'))
        # init an instance of this class
        estimator = Syn2Vec(**params)
        estimator.init_graph()
        # create a session
        estimator.sess = tf.Session(graph=estimator.graph)
        estimator.restore_from_saver(path)
        # evaluate the Variable normalized_embeddings and bind to final_embeddings
        estimator.final_embeddings = estimator.sess.run(estimator.normalized_embeddings)
        # bind dictionaries
        estimator.dictionary = json.load(open(os.path.join(path_dir, 'model_dict.json'), 'rb'))
        reverse_dictionary = json.load(open(os.path.join(path_dir, 'model_rdict.json'), 'rb'))
        # convert indices loaded from json back to int since json does not allow int as keys
        estimator.reverse_dictionary = {int(key): val for key, val in reverse_dictionary.items()}

        return estimator

    def sort(self, word):
        '''
        Use an input word to sort words using cosine distance in ascending order
        '''
        assert word in self.dictionary
        i = self.dictionary[word]
        vec = self.final_embeddings[i].reshape(1, -1)
        # Calculate pairwise cosine distance and flatten to 1-d
        pdist = pairwise_distances(self.final_embeddings, vec, metric='cosine').ravel()
        slist = []
        for i in pdist.argsort():
            if not (i in self.reverse_dictionary):
                print('Not in rdic i=%d' % i)
            else:
                slist.append(self.reverse_dictionary[i])
        return slist
        #return [self.reverse_dictionary[i] for i in pdist.argsort()]

    def transform(self, words):
        '''
        Look up embedding vectors using indices
        words: list of words
        '''
        # make sure all word index are in range
        # try:
        indices = []
        for w in words:
            if w in self.dictionary:
                indices.append(self.dictionary[w])
            else:
                print ('Word=%s not in dictionary' %w)
        # except KeyError:
        #     raise KeyError('Some word(s) not in dictionary')
        # else:
        return self.final_embeddings[indices]

if __name__ == '__main__':
    vocabulary_size = 500
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.
    valid_size = 16       # Random set of words to evaluate similarity on.
    valid_window = 100    # Only pick dev samples in the head of the distribution.
    n_neg_samples = 64    # Number of negative examples to sample.
    #n_steps = 2001
    n_steps = 100001
    learning_rate = 0.5

    save_path =  'models/text8'
    filename = 'data/text8_small.zip'
    words = read_data(filename)
    print('Data size', len(words))
    s2v = Syn2Vec(batch_size=batch_size, num_skips=num_skips, skip_window=skip_window,
             #architecture='cbow',
             architecture='skip-gram',
             embedding_size=embedding_size, vocabulary_size=vocabulary_size,
             loss_type='nce_loss',
             #loss_type='sampled_softmax_loss',
             n_neg_samples=n_neg_samples,
             #optimize='Adagrad',
             optimize='SGD',
             learning_rate=learning_rate, n_steps=n_steps,
             valid_size=valid_size, valid_window=valid_window, save_path=save_path)

    print(s2v.get_params())
    s2v.fit(words)
    del words  # Hint to reduce memory.
