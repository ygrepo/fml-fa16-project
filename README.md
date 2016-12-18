# FML-FA16-Project
Fundations of Machine Learning class final project

## Project Abstract

Improving quality of features vectors in vector embedding models by using synsets

### Methodology:

1. We use different NLP tools to generate synsets from a corpus (text8, Wiki500M-2016-dump).

2. The generated synset corpus is used in word2vec to obtain synset embedding vectors.

3. We evaluate the synset model accuracy using a synset version of Google's question-answer (19,558 questions)


## SetUp:

* The python script wordnet_utils.py perfoms text processing and it is in the syn2vec folder. wordnet_utils provides two modes:
    - SynsetStreamGenerator: processes a stream of words like the text8 file [1]: http://mattmahoney.net/dc/text8.zip
     to generate a stream of pairs (lemma, PartOfSpeech)
    - SynsetLineGenerator: processes line by line the input file (lquestions-words.txt present in the folder gold-data)
    to generate a new file of pairs (lemma,PartOfSpeech)
 The lemmatizer is WordnetLemmatizer from nltk [2]: http://www.nltk.org/.
 The default tagger is the "Perceptron Tagger" [3]: http://spacy.io/blog/part-of-speech-POS-tagger-in-python/,
 but can be switched easily to other taggers from nltk library.

* The Java programm WSD provides word disambiguation. It uses the library DKPro WSD [4]: https://dkpro.github.io/dkpro-wsd/.
DKPro gives the sensekey of Wordnet when disambiguating a pair (lemma, POS). Similarly to wordnet_utils, WSD provides
processing of:
 - stream of tuples
 - line of tuples

Training of the word or synset models are provided by two python scripts in the syn2vec folder:
* syn2vec.py a wrapper of word2vec, offering the following features:
  - cbow or skip-gram methodologies
  - nce_loss and sampled_softmax_loss loss functions
  - Adagrad, Adam and StochsticGradientDescent optimizers
  - loading, saving a model
  - tsne plotting

* word2vec_optimized.py which is a slightly customized version of the tensorflow code available at
 [5]:https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec_optimized.py, it
 allows to save the correct and incorrect predictions and other minor features.

 Other important folders:
 - **gold-data**: contains the data for evaluation, and the overall or per category results.
    Interesting files are:
      - 20160-12-14-global-results.txt
      - 2016-12-14-categories-results.txt
      - words-nearby.txt
      - words-synsets.csv
      - synsets-nearby.tx
      - synsets-words.csv

 - **deliverables**: contains the abstract of the project, the report, the queries used for WordNet database, and plots.
 - **scripts**: the different scripts to perform the transformations of the initial corpus, the training of the model,
   and the model evaluation:
    - generate-streamfiles.sh, generate-linefiles.sh: transforms the initial corpus into pairs of (lemma, POS)
    - map-streamfiles.sh, map-linefiles.sh: disambiguates the pairs of (lemma,POS) into sensekey
    - train-words.sh, train-synsets.sh: trains either the word or synset models
    - eval-words-quest-words.sh, eval-words-categories,sh, eval-synsets-quest-words.sh, eval-synsets-categories.sh:
      different evaluation scripts against Goolge's questions-words.txt in gold-data

