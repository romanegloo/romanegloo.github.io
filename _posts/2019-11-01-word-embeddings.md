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

![Word vectors in 2-dimensional; similar words have similar
vectors](http://me.jiho.us/images/posts/word-vectors-2d.png)

# References

* Y. Bengio, R. Ducharme, P. Vincent, and C. Jauvin, "A neural probabilistic
  language model," Journal of machine learning research, vol. 3, no. Feb, pp.
  1137–1155, 2003.
  \[[pdf](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)\]
