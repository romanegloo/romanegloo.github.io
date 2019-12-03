---
layout: post
title: "Word Embeddings"
date: 2019-11-01
---

With the rise of the Deep Learning paradigm, the methodology in NLP tasks has
gone through a huge transition; Most of the recent research efforts are in deep
neural network-based approaches with which replaced rather complicated feature
engineering methods. The use of deep learning methods in NLP tasks requires a
computational way of representing words semantically. Hence, such models start
with some kind of word representations (embeddings) at the bottom of its
architecture.

Word2vec by Mikolov et al. is one of the first word embeddings which gained huge
popularity in NLP and other related communities. Since the work of Word2vec,
numerous kinds of word embeddings have been proposed. Though all the word
embeddings have the same purpose of representing words in vector format, there
are different types of word embeddings depending on what kind of information is
used for training the representations. The unit of information can be at the
level of characters, words, or sequence of words in language models. You can
also utilize other hierarchical knowledge information of words or entities from
external sources.  In this post, I will review the methods of training word
embeddings starting with Word2vec.

# Introduction

## Word Embeddings

The idea of word embeddings is first proposed by [Bengio et
al.](http://me.jiho.us/bookmarks.html), and the gist of it has not been changed
drastically until now; We encode words into fixed-length vector representations
which can be used to solve many computational linguistic problems such as
question answering, text classfication, language translations, and so on. This
encoding model is based on the distributional hypothesis: linguistic items with
similar distributions in the vector space have similar meanings.

![Word vectors in 2-dimensional](https://me.jiho.us/images/posts/word-vectors-2d.png)  
*Figure. Word vectors in 2-dimensional; similar words have similar vectors 
([image source](http://suriyadeepan.github.io))*

Now, the distribution of words in the vector space allows us to deliver the
semantics of words by means of computing the relationships (e.g., difference)
between their vectors. For example, alalogy completion tasks can be done
utilizing the vectors of words, such as *pneumonia* to *lung* with *ischemic
colitis* to *colon*.

## Methodology

The methods of buliding word embeddings can be classified into two groups: (1)
**static word embeddings** in which the same word will always have the same
representation regardless of the context where the word occurs in, and (2)
**contextualised word embeddings** in which the context of a word is also taken
into consideration to build the representation. 

In this post, specific examples of these techniques will be introduced, which
include the followings:

* *static word embeddings*
    * Word2vec
    * CharCNN
    * fastText
* *contextualised word embeddings*
    * Elmo
    * BERT

# Static Word Embeddings

## Word2vec

*original paper: 
[Efficient Estimation of Word Representations in Vector Space (2013)](https://arxiv.org/pdf/1301.3781.pdf)*

*Word2vec* has been arguably the most popular word embeddings which made an
extensive influence on the field. *Word2vec* has two variant models that depend
on the target of the probabilities: *Skip-gram* and *CBOW*. These models are
implemented in a shallow neural network consisting of an input layer, a linear
projection layer, and an output score layer. The input layer is a one-hot input
vector representing the target word, the projection layer contains the dense
center vector of the word, and the output layer contains the scores which can
be interpreted as a probability distribution of words that are likely to be
seen in the target word’s context.

![CBOW and Skip-gram Model](https://me.jiho.us/images/posts/cbow-skipgram.png)  
*Figure. CBOW and Skip-gram Model*

The goal of *Skip-gram* model is to predict context words by the given target
word. The following negative log-likelihood loss function computes how the
predicted context words are close to the observed list of words.

$$ L = \sum_{c \in C_t} \log (1 + e^{-s_c}) + \sum_{n \in N_{t,c}} \log (1 + e^{s_n}), $$

where the first term adds the loss of the context words and the second term
adds the loss of the negative samples from the unigram distribution.

One limitation of this method is that this model can't handle OOV
(out-of-vocabulary) words, which have not occurred in a training dataset. The
following two methods, *CharCNN* and *fastText*, address this issue by
considering sub-word information in building word representations.

## CharCNN

*original paper: 
[Character-level convolutional networks for text classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)*

The authors of the *CharCNN* model claims that words can be represented by the
use of character-level convolutional networks (ConvNets) for NLP tasks. The
input sequence of words is mapped into the representations for its constituent
characters. They fixed the set of characters to the union of 26 English
characters, 10 digits, and 33 special characters. This model proved its strength
in representing OOV words, misspelled words, new words, and emoticons. It also
reduced model complexity by using a relatively small number of vector
representations. Following figure illustrates the CharCNN model.

![CharCNN Model](https://me.jiho.us/images/posts/CharCNN.png)  
*Figure. Character-level Convolutional Networks*

## fastText

*original paper:
[Enriching word vectors with subword information](https://www.mitpressjournals.org/doi/pdfplus/10.1162/tacl_a_00051)*

*fastText* took a further step by constructing word representations by its
constituent character-level n-grams.  Specifically to speak, the model
represents a word as the sum of its character n-grams representations, plus a
special boundary symbols at the beginning and end of words, plus the word
itself in the set of its n-grams. For example, a word *fast* is composed of the
following five n-grams, where n is set to 3:

\<fa, fas, ast, st\>, \<fast\>

With this model, grammatical variations that share most of n-grams, and
compound nouns are easy to model. As to the rest, it shares the same
architecture of the Word2vec Skip-gram model.

* [Download fastText pre-trained model](https://fasttext.cc/)

# Contextualised Word Embeddings

One of the issues of static word embeddings is that the word vectors are fixed
after training, which means that there can be only one fixed meaning of a word.
Hence, these representations cannot handle the
[polysemy](https://en.wikipedia.org/wiki/Polysemy) of natural languages. For
example, the representative meaning of *bank* can be only one (whatever that is
trained to) within the static word embeddings, nevertherless this word can have
multiple senses; bank as a financial institute or a river bank. Following
contextualised word embeddings are the attempts to address this issue in static
word embeddings.

## ELMo

*original papar:
[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)*

ELMo ("Embeddings from Language MOdels") is a deep contextualized word
representation. ELMo models the characteristics of word use (syntax and
semantics) and polysemy (different uses of words across linguistics contexts).
These word vectors are computed from the internal states of a two layers
bidirectional language model (biLM), which is pre-trained on a large text
corpus. Different layers of the language models encode different linguistic
aspects on the word; For example, Part-Of-Speech is better encoded by the
lower-level layers, while word-sense disambiguation is better predicted in the
higher-level layers.

![Model Architecture of TagLM](https://me.jiho.us/images/posts/taglm.png")

The above figure illustrates 

# References

* Y. Bengio, R. Ducharme, P. Vincent, and C. Jauvin, "A neural probabilistic
  language model," Journal of machine learning research, vol. 3, no. Feb, pp.
  1137–1155, 2003.
  \[[pdf](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)\]
* Mikolov, Tomas, et al. "Efficient estimation of word representations in
  vector space." arXiv preprint arXiv:1301.3781 (2013).
  \[[pdf](https://arxiv.org/pdf/1301.3781.pdf)\]
* Zhang, Xiang, Junbo Zhao, and Yann LeCun. "Character-level convolutional
  networks for text classification." Advances in neural information processing
  systems. 2015.
  \[[pdf](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)\]
* Bojanowski, Piotr, et al. "Enriching word vectors with subword information."
  Transactions of the Association for Computational Linguistics 5 (2017):
  135-146. 
  \[[pdf](https://www.mitpressjournals.org/doi/pdfplus/10.1162/tacl_a_00051)\]
