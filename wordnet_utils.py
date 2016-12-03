from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import os.path
import pickle
import zipfile

from itertools import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger
from nltk.wsd import lesk

# from pywsd.lesk import adapted_lesk

synsetid_synset_name_d_filename = 'synsetid_synset_name_d.p'
synset_name_synsetid_d_filename = 'synset_name_synsetid_d.p'
lemma_synsetid_d_filename = 'lemma_synsetid_d.p'
synsetid_lemma_d_filename = 'synsetid_lemma_d.p'

# sentences = ["the cat likes milk"]

sentences = ["the quick brown fox jumped over the lazy dog",
             "I love cats and dogs",
             "we all love cats and dogs",
             "cats and dogs are great",
             "sung likes cats",
             "she loves dogs",
             "cats can be very independent",
             "cats are great companions when they want to be",
             "cats are playful",
             "cats are natural hunters",
             "It's raining cats and dogs",
             "dogs and cats love sung"]

class TAG_MODE:
    FULL_TAGGING = 0
    ADJ = 1
    VERB = 2

class WordnetUtils():
    def __init__(self, path, win_size=3):
        self.path = path
        self.win_size = win_size
        self.create_synsets_dictionaries()
        self.create_lemma_unique_synset_dictionary()
        self.synsetid_synsetname_d = None
        self.synsetname_synsetid_d = None
        self.lemma_synsetid_d = None
        self.synsetid_lemma_d = None
        self.word_lemma_d = None
        self.tagger = PerceptronTagger()
        self.lemmatizer = WordNetLemmatizer()

    def create_synsets_dictionaries(self):
        '''
        From wordnet creates two dictionaries:
            synsetid -> synset name for ex. 2084071 -> 'dog.n.01'
            synset name -> synsetid  'dog.n.01' -> 2084071
        And pickle them to the disk
        '''
        f = '%s%s' % (self.path, synsetid_synset_name_d_filename)
        if not os.path.isfile(f):
            print("Generating synsetid to synset name dictionary")
            syns = list(wn.all_synsets())
            print('#of senses=' + str(len(syns)))
            offsets_list = [(s.offset(), s.name()) for s in syns]
            pdict = dict(offsets_list)
            pickle.dump(pdict, open(f, "wb"))
        else:
            print("Synset id to synset name dictionary already existing, skipping...")
            pdict = self.load_dict(synsetid_synset_name_d_filename)

        f = '%s%s' % (self.path, synset_name_synsetid_d_filename)
        if not os.path.isfile(f):
            print("Generating synset name to synset id dictionary")
            reverse_dictionary = dict(zip(pdict.values(), pdict.keys()))
            pickle.dump(reverse_dictionary, open(f, "wb"))
        else:
            print("Synset name to synset id dictionary already existing, skipping...")

    def create_lemma_unique_synset_dictionary(self):
        '''
        From wordnet create a dictionary which keeps track all the lemmas which are not ambiguous,
        i.e. one lemma has only one synsetid.
        And create the reverse dictionary and save it on disk.
        For example:
        pingpong_table ->  4381587
        4381587 -> pingpong_table
        '''
        f = '%s%s' % (self.path, lemma_synsetid_d_filename)
        if not os.path.isfile(f):
            print("Generating lemma to synset id dictionary")
            lemmas_in_wordnet = set(chain(*[ss.lemma_names() for ss in wn.all_synsets()]))
            dictionary = dict()
            for lemma in lemmas_in_wordnet:
                synsets = wn.synsets(lemma)
                if len(synsets) == 1:
                    print("Lemma=%s, synset=%s" % (lemma, synsets))
                    dictionary[lemma] = synsets[0].offset()
            print('Size of lemma uniqe synsets=' + str(len(dictionary)))
            pickle.dump(dictionary, open(f, "wb"))
        else:
            print("lemma to synset id dictionary already existing, skipping...")
            dictionary = self.load_dict(lemma_synsetid_d_filename)
        f = '%s%s' % (self.path, synsetid_lemma_d_filename)
        if not os.path.isfile(f):
            print("Generating synset id to lemma dictionary")
            reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
            pickle.dump(reverse_dictionary, open(f, "wb"))
        else:
            print("synset id to lemma dictionary already existing, skipping...")

    def lookup_synset_name_from_synsetid(self, synsetid):
        '''
        If not loaded, loads synsetid_synset_name dictionary
        Then given a synsetid returns its name.
        :param synsetid:
        :returns part_name
        '''
        if self.synsetid_synsetname_d == None:
            self.synsetid_synsetname_d = self.load_dict(synsetid_synset_name_d_filename)
        if synsetid in self.synsetid_synsetname_d:
            name = self.synsetid_synsetname_d[synsetid]
            if name is None:
                return synsetid
            else:
                part_name = name.split('.')[0]
                return part_name
        else:
            return synsetid

    def lookup_synsetid_from_synset_name(self, name):
        '''
        If not loaded, loads synset_name_synsetid dictionary
        Then given a synset name returns its synsetid
        :param name:
        :returns synsetid
        '''
        if self.synsetname_synsetid_d == None:
            self.synsetname_synsetid_d = self.load_dict(synset_name_synsetid_d_filename)
        if name in self.synsetname_synsetid_d:
            synsetid = self.synsetname_synsetid_d[name]
            return synsetid
        return None

    def lookup_lemma_from_synsetid(self, synsetid):
        '''
        Given a synsetid, lookup its lemma
        :param synsetid:
        :returns lemma
        '''
        if self.synsetid_lemma_d == None:
            self.synsetid_lemma_d = self.load_dict(synsetid_lemma_d_filename)
        if synsetid in self.synsetid_lemma_d:
            synsetid = self.synsetid_lemma_d[synsetid]
            return synsetid
        return None

    def lookup_synsetid_from_lemma(self, lemma):
        '''
        Given a lemma, lookup its synset id
        :param lemma:
        :returns synsetid
        '''
        if self.lemma_synsetid_d == None:
            self.lemma_synsetid_d = self.load_dict(lemma_synsetid_d_filename)
        if lemma in self.lemma_synsetid_d:
            synsetid = self.lemma_synsetid_d[lemma]
            return synsetid
        return None

    def create_word_to_lemma_dictionary(self, words):
        '''
        Given a list of words, create word_lemma dictionary in order to speed up mapping of a word to a synsetid.
        :param words:
        '''
        tags = self.tagger.tag(words)
        print(tags[:5])
        word_lemma_d = dict()
        for word, tag in tags:
            pos = get_wordnet_pos(tag)
            #            print("word=%s,tag=%s,pos=%s" %(word, tag, pos))
            if pos is not None:
                lemma = self.lemmatizer.lemmatize(word, pos)
                if lemma is not None:
                    #                    print(lemma)
                    word_lemma_d[word] = lemma
                else:
                    print("Lemma not found for word=" + word)
        print("Word_lemma_dictionary size=%s" % (len(word_lemma_d.items())))
        return word_lemma_d

    def lookup_lemma_from_word(self, words, word):
        '''
        Given a word, lookup its lemma
        :param word:
        :returns lemma
        '''
        if self.word_lemma_d == None:
            self.word_lemma_d = self.create_word_to_lemma_dictionary(words)
        if word in self.word_lemma_d:
            lemma = self.word_lemma_d[word]
            return lemma
        return None

    def clean_data(self, words):
        '''
        Given a list of words, remove the stop words and returns the most common words
        :param filename:
        :param lemma:
        :returns filtered_words
        '''
        stops = set(stopwords.words("english"))
        filtered_words = [word for word in words if word not in stops]
        print(filtered_words[:10])
        count = collections.Counter(filtered_words).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        for word in words:
            if word in dictionary:
                data.append(word)
        return data

    def load_dict(self, filename):
        '''
        Loads senses to names dictionary.
        :param filename
        :returns senses_names_d
         '''
        f = '%s%s' % (self.path, filename)
        senses_names_d = pickle.load(open(f, "rb"))
        return senses_names_d

    def save_synsetids(self, filename, words):
        '''
        Generates senses to name dictionary and pickle them.
        :param filename
        :param words
         '''
        synsetids = self.generate_synsetids(words)
        print("Synsetids", synsetids[:5])
        pickle.dump(synsetids, open(filename, "wb"))

    def generate_synsetids(self, words):
        '''
        This function generates a list of synsetids for a given list of words by mapping each word to its synsetid.
        Given a list of words:
        1. Removes stop words and keeps most common words
        2. for each word:
           - finds the lemma
           - using its lemma find the synsetid
           - if there is no synsetId then try to disambiguate the word using lesk
           - It is slow to disambuguate a larger corpus so only a window around the word to disambiguate is given
             as context of disambiguation
           - if we have a synset then add it to the list of synsetids if not just add the ambiguous word
        :param words:
        :return: synsetids
        '''
        filtered_words = self.clean_data(words)
        print("Words", filtered_words[:5])
        unknown_count = 0
        synsetids = []
        i = 0
        words_sz = len(words)
        for word in words:
            lemma = self.lookup_lemma_from_word(words, word)
            if lemma is None:
                unknown_count += 1
                synsetid = word
            else:
                synsetid = self.lookup_synsetid_from_lemma(lemma)
                if synsetid is None:
                    # synset = lesk(sentence, word)
                    (li, ri) = get_window_indices(words_sz, i, self.win_size)
                    win_words = words[li: i] + words[i: ri + 1]
                    # print(win_words)
                    synset = lesk(win_words, word)
                    if synset is None:
                        unknown_count += 1
                        synsetid = word
                    else:
                        synsetid = synset.offset()
            if synsetid == word:
                print("No synsetid for=", word)
            else:
                print("word=%s, synsetName=%s, synsetId=%d" % (
                word, self.lookup_synset_name_from_synsetid(synsetid), synsetid))
            synsetids.append(synsetid)
            i += 1
        synsetids_size = len(synsetids)
        print(synsetids_size)
        if synsetids_size != 0:
            rejected_percent = 100.0 * (unknown_count / synsetids_size)
        else:
            rejected_percent = 0
        print("Synsetids_size list size=%d, rejected=%.3f" % (synsetids_size, rejected_percent))
        return synsetids

    def generate_lemma_pos_with_win(self, words):
        '''
        This function generates lemma to POS using the selected tagger (by default Perceptron Tager)
        and a lemmatizer to find given a word its lemma and POS using a context window.
        It returns a list of (lemma, POS).
        :param words:
        :return: tuples
        '''
        filtered_words = self.clean_data(words)
        print("Filtered words", filtered_words[:5])
        tuples = []
        words_sz = len(filtered_words)
        print("Processing %d words" % words_sz)
        i = 0
        unknown_count = 0
        for word in filtered_words:
            (li, ri) = get_window_indices(words_sz, i, self.win_size)
            win_words = filtered_words[li: i] + filtered_words[i: ri + 1]
            print("Word=%s, win_words=%s" %(word,win_words))
            tags = self.tagger.tag(win_words)
            print ('Tags=%s' %tags[:5])
            word_tag_d = dict(tags)
            w_tag = word_tag_d[word]
            if w_tag is None:
                print("No tag for word=%s" % word)
                i += 1
                unknown_count += 1
                tuples.append((word, None))
                continue
            pos = get_wordnet_pos(w_tag)
            if pos is None:
                print("No pos for word=%s" % word)
                i += 1
                unknown_count += 1
                tuples.append((word, None))
                continue
            # print("word=%s,tag=%s,pos=%s" %(word, w_tag, pos))
            lemma = self.lemmatizer.lemmatize(word, pos)
            # print("Lemma=%s" %lemma)
            tuples.append((lemma, pos))
            # synset = lesk(win_words, lemma, pos)
            # print("Synset=%s" %synset)
            i += 1
            print('i=%d' % i)
        if words_sz != 0:
            rejected_percent = 100.0 * (unknown_count / words_sz)
        else:
            rejected_percent = 0
        print("word_size=%d, rejected=%.3f" % (words_sz, rejected_percent))
        return tuples

    def generate_lemma_pos(self, words):
        '''
        This is a similar function than the one above without contextual window.
        :param words:
        :return: tuples
        '''
        words_sz = len(words)
        tags = self.tagger.tag(words)
        #print('Tags=%s' % tags[:5])
        i = 0
        tuples = []
        unknown_count = 0
        for word, tag in tags:
            pos = get_wordnet_pos(tag)
            if pos is None:
                #print("No pos for word=%s" % word)
                i += 1
                unknown_count += 1
                continue
            lemma = self.lemmatizer.lemmatize(word, pos)
            #print("word=%s,lemma=%s,tag=%s,pos=%s" % (word, lemma, tag, pos))
            tuples.append((lemma, pos))
            i += 1
        if words_sz != 0:
            rejected_percent = 100.0 * (unknown_count / words_sz)
        else:
            rejected_percent = 0
        #print("word_size=%d, rejected=%.3f" % (words_sz, rejected_percent))
        return tuples

    def generate_lemma_addj(self, words):
        '''
       This helper function is used for the questions-answers challenge.
       The tagger always gets wrongs when the question consists of 3 adjectives
       thus generates for the 3 lemme the POS 'a'.
       It returns a list of (lemma, 'a').
       :param words:
       :return: tuples
       '''
        tuples = []
        for word in words:
            lemma = self.lemmatizer.lemmatize(word, 'a')
            #print("word=%s,lemma=%s" % (word, lemma))
            tuples.append((lemma, 'a'))
        return tuples

    def generate_lemma_verb(self, words):
        '''
       This helper function is used for the questions-answers challenge.
       The tagger always gets wrongs when the question consists of 3 verbs
       thus generates for the 3 lemme the POS 'v'.
       It returns a list of (lemma, 'v').
       :param words:
       :return: tuples
       '''
        tuples = []
        for word in words:
            lemma = self.lemmatizer.lemmatize(word, 'v')
            #print("word=%s,lemma=%s" % (word, lemma))
            tuples.append((lemma, 'v'))
        return tuples

    def transform(self, synsetids):
        words = []
        for s in synsetids:
            w = self.lookup_synset_name_from_synsetid(s)
            # if w is None:
            words.append(str(w))
            # else:
            #     words.append(w)
        return words


def get_sample_words():
    # sentences to words
    words = " ".join(sentences).split()
    return words

# Read the data into a list of strings.
def read_words(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data

def read_unzipped_words(filename):
    with open(filename, 'r') as thefile:
        data= thefile.read()
    words = data.split()
    return words

def read_lines(filename):
    data = open(filename)
    dat = data.read()
    lst = dat.splitlines()
    return lst

def restore_synsetids(filename):
    synsetids = pickle.load(open(filename, "rb"))
    return synsetids

def save_to_text_file(output_f, tuples):
    total = len(tuples)
    filename = '%s%s' % (path, output_f)
    print("Processing tuples=%d" % total)
    i = 0
    with open(filename, 'w') as f:
        for (lemma, tag) in tuples:
            print("lemma=%s, tag=%s" %(lemma, tag))
            token = ' %s,,%s\\n' % (lemma, tag)
            f.write(' ' + token)
            i += 1
            print("Processing i=%d,total=%d" % (i, total))

def save_tuples_to_file(output_f, tuples):
    total = len(tuples)
    filename = '%s%s' % (path, output_f)
    print("Processing tuples=%d" % total)
    i = 0
    with open(filename, 'w') as f:
        for (lemma, tag) in tuples:
            #print("lemma=%s, tag=%s" %(lemma, tag))
            token = ' %s,,%s' % (lemma, tag)
            f.write(' ' + token)
            i += 1
            print("Processing i=%d,total=%d" % (i, total))

def save_list_list_to_text_file(output_f, tuples):
    total = len(tuples)
    filename = '%s%s' % (path, output_f)
    print("Processing list of tuples=%d" % total)
    i = 1
    with open(filename, 'w') as f:
        for tuples_l in tuples:
            if len(tuples_l) == 0:
                i += 1
                continue
            felm = tuples_l[0]
            if isinstance(felm, str) and felm.startswith(":"):
                f.write(felm  + '\n')
                i += 1
                continue
            print(tuples_l)
            line = ""
            for (lemma, tag) in tuples_l:
                token = ' %s,,%s' % (lemma, tag)
                line += token
            line += '\n'
            print("Processed i=%d,total=%d" % (i, total))
            #print("line=%s" %line)
            f.write(line)
            i += 1

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def get_window_indices(l, i, ws):
    li = i - ws
    if li < 0:
        li = 0
    ri = i + ws
    if ri >= (l - 1):
        ri = l - 1
    return (li, ri)


def test_window():
    words = get_sample_words()
    print("Initial words")
    print(words)
    i = 0
    words_sz = len(words)
    winsz = 4
    for word in words:
        print("Target word=%s" % word)
        (li, ri) = get_window_indices(words_sz, i, winsz)
        sentence = words[li: i] + words[i: ri + 1]
        print(sentence)
        i += 1


def test_lesk():
    words = get_sample_words()
    print("Words =%s" % words)
    tagger = PerceptronTagger()
    tags = tagger.tag(words)
    print('Tags=%s' % tags[:5])
    lemmatizer = WordNetLemmatizer()
    for word, tag in tags:
        pos = get_wordnet_pos(tag)
        if pos is None:
            continue
        # print("word=%s,tag=%s,pos=%s" %(word, tag, pos))
        lemma = lemmatizer.lemmatize(word, pos)
        # print("Lemma=%s" %lemma)
        synset = lesk(words, lemma, pos)
        if synset is None:
            print('No synsetid for word=%s' % word)
        else:
            print('word=%s, synsetname=%s, synsetid=%d' % (word, synset.name(), synset.offset()))


# def test_pywsd():
#     print("#TESTING adapted_lesk() ...")
#     for i in range(len(sentences)):
#         sentence = sentences[i]
#         print(sentence)
#         words = sentence.split()
#         for word in words:
#             synset = adapted_lesk(sentence,word)
#             if synset is None:
#                 print('No synsetid for word=%s' %word)
#             else:
#                 print('word=%s, synsetname=%s, synsetid=%d' %(word,synset.name(), synset.offset()))

def test_dictionaries(path):
    wn_utils = WordnetUtils(path)
    dog = wn.synset('dog.n.01')
    print(dog.name(), dog.offset())
    name = wn_utils.lookup_synset_name_from_synsetid(dog.offset())
    print("Lookup dog name using dog synset id, got name=" + name)
    synsetid = wn_utils.lookup_synsetid_from_synset_name(dog.name())
    print("Lookup dog synset id given dog name, id=" + str(synsetid))
    pingpong_table = wn.synset('table-tennis_table.n.01')
    print(pingpong_table.name(), pingpong_table.offset())
    lemma = wn_utils.lookup_lemma_from_synsetid(pingpong_table.offset())
    print("Lookup lemma from synset id, got lemma=" + lemma)
    lemma = 'pingpong_table'
    synsetid = wn_utils.lookup_synsetid_from_lemma(lemma)
    print("Lookup synset id from lemma, got synsetid=" + str(synsetid))


def sample(path):
    words = get_sample_words()
    print("Initial words")
    print(words)
    wn_utils = WordnetUtils(path)
    filename = '%s%s' % (path, 'sample_words_synsets_d.p')
    # wn_utils.save_synsetids(filename, words)
    synsetids = restore_synsetids(filename)
    words = wn_utils.transform(synsetids)
    print("Restored words")
    print(words)


def generate_text8_words_synsets(path):
    '''
    Generate synsets using nltk lek from text8 file.
    '''
    filename = '%s%s' % (path, 'text8.zip')
    # filename = '%s%s' % (path, 'text8_small.zip')
    words = read_words(filename)
    print("Initial words")
    print(words[:10])
    wn_utils = WordnetUtils(path)
    filename = '%s%s' % (path, 'text8_words_synsets_d.p')
    # filename = '%s%s' % (path, 'text8_small_words_synsets_d.p')
    wn_utils.save_synsetids(filename, words)
    # synsetids = restore_synsetids(filename)
    # words = wn_utils.transform(synsetids)
    # print("Restored words")
    # print(words[:10])


def generate_text8_lemma_pos(path):
    '''
    Generate a list of tuples (lemma,POS) from text8
    '''
    filename = '%s%s' % (path, 'text8.zip')
    words = read_words(filename)
    # print("Initial words")
    # print(words[:10])
    wn_utils = WordnetUtils(path)
    tuplel = wn_utils.generate_lemma_pos_with_win(words)
    # filename = '%s%s' % (path, 'text8_l_pos.p')
    # pickle.dump(tuplel, open(filename, "wb" ))
    #save_to_text_file('text8-l-os.txt', tuples)

def generate_news_lemma_pos(path):
    wn_utils = WordnetUtils(path)
    filename = '%s%s' % (path, 'news.2010.en.shuffled.clean')
    word_list = read_lines(filename)
    print("word_list=%s" % word_list[:10])
    #word_list = word_list[:10]
    tuples = []
    for word_line in word_list:
        words = word_line.split()
        tuplel = wn_utils.generate_lemma_pos(words)
        tuples += tuplel
    print("Tuples=%s" %tuples[:10])
    save_tuples_to_file('news-2010-en-shuffled-l-pos.txt', tuples)


def determine_tagging_mode(word_line):
    if word_line.startswith(": gram1"):
        return TAG_MODE.FULL_TAGGING
    elif word_line.startswith(": gram2"):
        return TAG_MODE.ADJ
    elif word_line.startswith(": gram3"):
        return TAG_MODE.ADJ
    elif word_line.startswith(": gram4"):
        return TAG_MODE.ADJ
    elif word_line.startswith(": gram5"):
        return TAG_MODE.VERB
    elif word_line.startswith(": gram6"):
        return TAG_MODE.FULL_TAGGING
    elif word_line.startswith(": gram7"):
        return TAG_MODE.VERB
    elif word_line.startswith(": gram8"):
        return TAG_MODE.FULL_TAGGING
    elif word_line.startswith(": gram9"):
        return TAG_MODE.VERB
    elif word_line.startswith(":"):
        return TAG_MODE.FULL_TAGGING
    return None

def generate_qa_lemma_pos(path):
    '''
    Generate a list of tuples (lemma,POS) from questions-words.txt.
    '''
    wn_utils = WordnetUtils(path)
    filename = '%s%s' % (path, 'capital-common-countries.txt')
    word_list = read_lines(filename)
    print("word_list=%s" % word_list[:10])
    tuples = []
    tagging_mode = TAG_MODE.FULL_TAGGING
    for word_line in word_list:
        v_tag_mode = determine_tagging_mode(word_line)
        if v_tag_mode is not None:
            tagging_mode = v_tag_mode
            print("Ignoring %s, tagging_mode=%s" %(word_line, tagging_mode))
            tuplel = [word_line]
            tuples.append(tuplel)
            continue
        words = word_line.split()
        if tagging_mode == TAG_MODE.FULL_TAGGING:
            tuplel = wn_utils.generate_lemma_pos(words)
        elif tagging_mode == TAG_MODE.ADJ:
            tuplel = wn_utils.generate_lemma_addj(words)
        elif tagging_mode == TAG_MODE.VERB:
            tuplel = wn_utils.generate_lemma_verb(words)

        #print("Tuples=%s" %tuplel)
        tuples.append(tuplel)
    print("Tuples=%s" %tuples[:10])
    save_list_list_to_text_file('capital-common-countries-l-pos.txt', tuples)


if __name__ == '__main__':
    path = 'data/'
    # test_dictionaries(path)
    # test_lesk()
    # test_pywsd()
    # test_window()
    # sample(path)
    # generate_text8_words_synsets(path)
    # generate_text8_lemma_pos(path)
    generate_qa_lemma_pos(path)
    # generate_news_lemma_pos(path)
