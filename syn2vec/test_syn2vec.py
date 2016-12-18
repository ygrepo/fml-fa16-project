import zipfile

from syn2vec import Syn2Vec
from syn2vec import build_dataset, generate_batch_skipgram, plot, plot_with_labels
from wordnet_utils import WordnetUtils
from wordnet_utils import read_stream_zip_file, read_stream_file, get_sample_words, restore_synsetids


def save_list_to_file(lst, filename):
    f = open(filename, 'w')
    with open(filename,'w') as f:
        for item in lst:
            st = str(item)
            f.write(' ' + st)
    print("Save synsets into file=%s" %filename)


def fit(words, n_steps=100001, save_path='../models/test_model'):
    vocabulary_size = 500
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.
    valid_size = 16       # Random set of words to evaluate similarity on.
    valid_window = 100    # Only pick dev samples in the head of the distribution.
    n_neg_samples = 64    # Number of negative examples to sample.
    n_steps = n_steps
    learning_rate = 0.5
    save_path = save_path
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

    print s2v.get_params()
    s2v.fit(words)
    print(s2v.final_embeddings.shape)

def restore(path):
    s2v = Syn2Vec.restore(path)
    print("Params")
    print(s2v.get_params())
    print("Embeddings dimension")
    print(s2v.final_embeddings.shape)
    print("Dictionary")
    print (s2v.dictionary.items()[:10])
    print("Reverse dictionary")
    print (s2v.reverse_dictionary.items()[:10])
    return s2v

def test_similarity(s2v, synsetid):
    similarl = s2v.sort(synsetid)
    print("Emb. length vector for word=%s, lenght=%d" % (synsetid, len(similarl)))
    print("Most similar to synset=%s" % synsetid)
    print(similarl[:10])
    return similarl


def test_text8():
    #filename = '../data/text8.zip'
    filename = '../data/2016-12-07-text8-synsets.txt'
    words = read_stream_file(filename)
    #words = words[:10000]
    save_path =  '../models/synsets'
  #  print('Data size', len(words))
    #fit(words, 100001, save_path)
    s2v = restore('%s%s' %(save_path, '/model.ckpt'))
    # test_similarity(s2v, 'religious')
    # print(s2v.transform(['religious']).shape)
    plot(s2v, '../deliverables/tsne-senses-text8.png')


def test_sample():
    save_path =  '../models/sample'
    # words = get_sample_words()
    # print('Data size', len(words))
    # fit(words, 2000, save_path)
    #s2v = restore('%s%s' %(save_path, '/model.ckpt'))
    #test_similarity(s2v, 'dogs')
    #print(s2v.transform(['cats']).shape)

def test_sample_synsets():
    path = '../data/'
    filename = '%s%s' % (path, 'text8_words_synsets.p')
    # synsetids = restore_synsetids(filename)
    # print("Reloaded=%d synsets" %len(synsetids))
    # print("Synsets")
    # print(synsetids[:20])
    wn_utils = WordnetUtils(path)
    synsetids = [2669789, 6613686, 5128519, 14649197]
    words = wn_utils.transform(synsetids)
    print("Restored words")
    print(words)
    save_path =  '../models/sample_synsets'
    #fit(synsetids, 10000, save_path)
    s2v = restore('%s%s' %(save_path, '../model.ckpt'))
    target = 5128519
    target_name = wn_utils.lookup_synset_name_from_synsetid(target)
    starget = str(target)
    synsetids = test_similarity(s2v, starget)
    words = wn_utils.transform(synsetids)
    print("Similar to synsetId=%d, word=%s" %(target, target_name))
    print(words[:10])

def test_save_synsets_to_file():
    path = '../data/'
    filename = '%s%s' % (path, 'text8_words_synsets.p')
    synsetids = restore_synsetids(filename)
    print("Reloaded=%d synsets" %len(synsetids))
    print("Synsets")
    print(synsetids[:20])
    filename = '%s%s' % (path, 'text8_words_synsets.txt')
    save_list_to_file(synsetids, filename)

#
if __name__ == '__main__':
    #test_sample()
    #test_generate_batch()
    test_text8()
    #test_sample_synsets()
    #test_save_synsets_to_file()


