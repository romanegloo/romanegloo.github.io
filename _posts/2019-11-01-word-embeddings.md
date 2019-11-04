---
layout: post
title: "Word Embeddings"
date: 2019-11-01
---

With the rise of the Deep Learning paradigm, the methodology in NLP tasks has gone
through a huge transition; Most of the recent research efforts are in deep
neural network-based approaches with which replaced rather complicated feature
engineering methods. The use of deep learning methods in NLP tasks requires a
computational way of representing words semantically. Hence, such models start
with some kind of word representations (embeddings) at the bottom of its
architecture.

Word2vec by Mikolov et al. is one of the first word embeddings which gained
huge popularity in NLP and other related communities. Since the work of
Word2vec, numerous kinds of word embeddings have been proposed. Though all
the word embeddings have the same purpose of representing words in vector
format, there are different types of word embeddings depending on what kind of
information is used for training the representations. The unit of information
can be at the level of characters, words, or sequence of words in language
models. You can also utilize other hierarchical knowledge information of words
or entities from external sources.  In this post, I will review the methods of
training word embeddings starting with Word2vec.

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
*Word vectors in 2-dimensional; similar words have similar vectors 
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

In this post, specific examples of these techniques will be introduced,
which include the followings:

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

![CBOW and Skip-gram Model](https://me.jiho.us/images/cbow-skipgram.png)
*CBOW and Skip-gram Model*

The goal of *Skip-gram* model is to predict context words by the given target
word. The following negative log-likelihood loss function computes how the
predicted context words are close to the observed list of words.

$$ L = \sum_{c \in C_t} \log(1 + e^{-s_c}) + \sum_{n \in N_{t,c}} \log(1+e^{s_n}), $$

# References

* Y. Bengio, R. Ducharme, P. Vincent, and C. Jauvin, "A neural probabilistic
  language model," Journal of machine learning research, vol. 3, no. Feb, pp.
  1137–1155, 2003.
  \[[pdf](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)\]
