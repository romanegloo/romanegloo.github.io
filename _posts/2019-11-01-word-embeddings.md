---
layout: post
title: "Word Embeddings"
date: 2019-11-01
---

# Introduction

With the rise of Deep Learning paradigm, the methodology in NLP tasks has gone
through a huge transition; Most of the recent research efforts are in deep
neural network-based approaches with which replaced rather complicated feature
engineering methods. The use of deep learning methods in NLP tasks requires a
computational way of representing words semantically. Hence, most of such
models start with some kind of word representations (embeddings) at the bottom
of its architecture.

Word2vec by Mikolov et al. is one of the first word embeddings which gained
huge popularity in NLP and other related communities. Since the work of
Word2vec, a numerous kinds of word embeddings have been proposed. Though, all
the word embeddings have the same purpose of representing words in vector
format, there are different types of word embeddings depending on what kind of
information is used for training the representations. The unit of information
can be at the level of characters, words, or sequence of words in language
models. You can also utilize other hierarchical knowledge information of words
or entities from external sources.  In this post, I will review the methods of
training word embeddings starting with Word2vec.


