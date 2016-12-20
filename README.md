# FML-FA16-Project
Fundations of Machine Learning class final project

## Project Abstract

Improving quality of features vectors in vector embedding models by using synsets

### Methodology:

1. We use different NLP tools to generate synsets from a corpus (text8, Wiki500M-2016-dump).

2. The generated synset corpus is used in word2vec to obtain synset embedding vectors.

3. We evaluate the synset model accuracy using a synset version of Google's question-answer (19,558 questions)

4. We also train a synset model on sentence classification task


## SetUp:

* The python script wordnet_utils.py perfoms text processing and it is in the syn2vec folder. wordnet_utils provides two modes:
    - 'SynsetStreamGenerator': processes a stream of words like the [text8] (http://mattmahoney.net/dc/text8.zip) file
     to generate a stream of pairs (lemma, PartOfSpeech)
    - 'SynsetLineGenerator': processes line by line the input file (lquestions-words.txt present in the folder gold-data)
    to generate a new file of pairs (lemma,PartOfSpeech)
 The lemmatizer is WordnetLemmatizer from [nltk] (http://www.nltk.org/).
 The default tagger is the [Perceptron Tagger] (http://spacy.io/blog/part-of-speech-POS-tagger-in-python/),
 but can be switched easily to other taggers from nltk library.

* The Java programm WSD provides word disambiguation. It uses the library [DKPro WSD] ('https://dkpro.github.io/dkpro-wsd/)
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

* word2vec_optimized.py which is a slightly customized version of the [tensorflow] (https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec_optimized.py) code:
 It allows to save the correct and incorrect predictions and other minor features.

The following files **are not present** and can be retrieved using these instructions:
+ `wget https://transfer.sh/dHdff/text8-synsets-model.ckpt`
+ `wget https://transfer.sh/oNhTg/text8-l-pos.txt`
+ `wget https://transfer.sh/3oids/2016-12-07-text8-synsets.txt`
+ `wget https://transfer.sh/7RlZL/text8-model.ckpt`
+ `wget https://transfer.sh/dHdff/text8-synsets-model.ckpt`
+ `wget https://transfer.sh/9fUoc/wsd.jar`

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
    - 'generate-streamfiles.sh, generate-linefiles.sh': transforms the initial corpus into pairs of (lemma, POS)
    - 'map-streamfiles.sh, map-linefiles.sh': disambiguates the pairs (lemma,POS) into sensekeys
    - 'train-words.sh, train-synsets.sh': trains either the word or synset models
    - 'eval-words-quest-words.sh, eval-words-categories,sh, eval-synsets-quest-words.sh, eval-synsets-categories.sh':
      different evaluation scripts against Goolge's questions-words.txt in gold-data

- **sentence-classification**: Includes the modified code from [harvardnlp-sent-conv-torch](https://github.com/harvardnlp/sent-conv-torch) repo to test the new embeddings on various sentence classification tasks.

     + `convertData.sh` is used to convert original data files presented in the original [sent-conv-torch/data](https://github.com/harvardnlp/sent-conv-torch/tree/master/data) repo to their synset-id versions. One can copy and rename original `/data` folder to `/dataold` and call `./convertData.sh dataold/*.{all,dev,test,train}`

     + Download text8-synset trained word embeddings with `wget https://transfer.sh/JtS2t/v8.bin`

     + Use `python preprocess.py MR v8.bin` to prepare hdf5 data for training.

     + Use `train.sh` to submit job to a TORQUE cluster or use `th main.lua -data MR.hdf5` command to train. Results are printed and saved in `results/` folder.

     + Use `-cudnn 1 -gpuid 1` flags to enable gpu



In addition besides python 2.7 and various libraries used in the python scripts like (numpy, panda: recommendation is
to install anaconda, nltk), the following have also to be installed: tensorflow framework, lua, torch.




