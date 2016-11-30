##Project Abstract

  Improving quality of features vectors in vector embedding models by using synsets  

  Word2vec is a popular software from Google available on the Tensorflow  framework which has  given existence to many papers. Word2vec is a shallow neural network with one hidden layer allowing one-hot encoding of the words. It uses a vector representation of the words, each word id (0 to the number of words in the vocabulary) is mapped to a low-dimensional vector-space from their distributional properties observed in the text corpus given as input to word2vec.  The output feature vectors are the probabilities that each word are embedded nearby each other (semantically similar). The weights or feature vectors are learnt using two methods: CBOW or skip-gram. Although different in nature they have both the same end results: vectors of words judged similar by their context are nudged closer together. For a faster training, these models instead of computing similarities of every words against every context, selects randomly contexts, a method known as negative sampling. Words have more than one meaning which is captured in a lexical database like WordNet. And on goal of this project is to improve Word2vec by using WordNet.

  At the core of wordnet is a synset, an unordered sets of synonyms which could be nouns, verbs or adjectives. A synset contains a definition, a gloss, and can have different types of relations with other synsets like ISA relation. The main idea  of  the project is to use word2vec in the synset feature space, the "tokens" used by word2vec are the synset ids. And given an input word2vec generate similarities between synset ids. With this new set of features vectors we evaluate the quality of our words representation on a word sense disambiguation evaluation task such as SenseEval and SemVal. 

  To generate the training data which consists of the sequence of synsets ids corresponding to the initial document, we use a tagged database like Babelnet, or a boosting algorithm with base classifier Lesk. We also compare the performance of the experiment using Word2vec and a combination of POS Tagger (word to synset id tag) and the energy based algorithm MaxEnt.

  Finally to evaluate the validity of synset id vectors as vector space, we reuse it in another word vector embeddings software GloVe which uses word-context matrices.

  Could we improve the cosine similarity by using a kernel?

Links 
http://wordnet.princeton.edu/wordnet/download/current-version/


