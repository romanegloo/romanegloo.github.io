---
layout: post
title: "Word Embeddings"
date: 2019-11-01
---

<!-- http://www.davidsbatista.net/blog/2018/12/06/Word_Embeddings/ -->


With the rise of the Deep Learning paradigm, the methodology in NLP tasks has
gone through a huge transition; Most of the recent research efforts are in
deep neural network-based approaches with which replaced rather complicated
feature engineering methods. The use of deep learning methods in NLP tasks
requires a computational way of representing words semantically. Hence, such
models start with some kind of word representations (embeddings) at the
bottom of its architecture.

Word2vec by Mikolov et al. is one of the first word embeddings which gained
huge popularity in NLP and other related communities. Since the work of
Word2vec, numerous kinds of word embeddings have been proposed. Though all
the word embeddings have the same purpose of representing words in vector
format, there are different types of word embeddings depending on what kind
of information is used for training the representations. The unit of
information can be at the level of characters, words, or sequence of words in
language models. You can also utilize other hierarchical knowledge
information of words or entities from external sources. In this post, I will
review the methods of training word embeddings starting with Word2vec.

# Introduction

## Word Embeddings

The idea of word embeddings is first proposed by [Bengio et
al.](http://me.jiho.us/bookmarks.html), and the gist of it has not been changed
drastically until now; We encode words into fixed-length vector representations
which can be used to solve many computational linguistic problems such as
question answering, text classfication, language translations, and so on. This
encoding model is based on the distributional hypothesis, "linguistic items with
similar distributions in the vector space have similar meanings."

<div class="text-center mb-5">
    <img src="https://me.jiho.us/images/posts/word-vectors-2d.png" 
         alt="Word vectors in 2-dimensional" class="img-fluid"/>
    <div class="caption center-block">
        Word vectors in 2-dimensional; Similar words have similar vectors
        ([<a href="http://suriyadeepan.github.io">image source</a>])
    </div>
</div>

Now, the distribution of words in the vector space allows us to deliver the
semantics of words by means of computing the relationships (e.g., difference)
between their vectors. For example, alalogy completion tasks can be done
utilizing the vectors of words, such as *pneumonia* to *lung* with *ischemic
colitis* to *colon*.

## Methodology

The methods of buliding word embeddings can be classified into two groups: (1)
**static word embeddings** in which the same word will always have the same
representation regardless of the context where the word occurs in, and (2)
**contextualized word embeddings** in which the context of a word is also taken
into consideration to build the representation. 

In this post, specific examples of these techniques will be introduced,
which include the followings:

* *Static Word Embeddings*
    * Word2vec
    * CharCNN
    * fastText
* *Contextualized Word Embeddings*
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

<div class="text-center mb-5">
    <img src="https://me.jiho.us/images/posts/cbow-skipgram.png" 
         alt="CBOW and Skip-gram Model" class="img-fluid"/>
    <div class="caption center-block">
        Figure. CBOW and Skip-gram Model
    </div>
</div>

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

The authors of the *CharCNN* model claims that words can be represented by
the use of character-level convolutional networks (ConvNets) for NLP tasks. The
input sequence of words is mapped into the representations for its constituent
characters. They fixed the set of characters to the union of 26 English
characters, 10 digits, and 33 special characters. This model proved its
strength in representing OOV words, misspelled words, new words, and emoticons.
It also reduced model complexity by using a relatively small number of vector
representations. Following figure illustrates the CharCNN model.

<div class="text-center mb-5">
  <img src="https://me.jiho.us/images/posts/CharCNN.png"
       class="img-fluid" alt="CharCNN Model"/>
  <div class="caption text-center">Figure. Character-level Convolutional Networks</div>
</div>

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

### External Links

* [Download fastText pre-trained model](https://fasttext.cc/)

# Contextualized Word Embeddings

One of the issues of static word embeddings is that the word vectors are fixed
after training, which means that there can be only one fixed meaning of a word.
Hence, these representations cannot handle the
[polysemy](https://en.wikipedia.org/wiki/Polysemy) of natural languages. For
example, the representative meaning of *bank* can be only one (whatever that is
trained to) within the static word embeddings, nevertherless this word can have
multiple senses; bank as a financial institute or a river bank. Following
contextualized word embeddings are the attempts to address this issue in static
word embeddings.

## ELMo

*original papar:
[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)*

ELMo ("Embeddings from Language MOdels") is a deep contextualized word
representation. ELMo models the characteristics of word use (syntax and
semantics) and polysemy (different uses of words across linguistics
contexts). These word vectors are computed from the internal states of a two
layers bidirectional language model (biLM), which is pre-trained on a large
text corpus. Different layers of the language models encode different
linguistic aspects on the word; For example, Part-Of-Speech is better encoded
by the lower-level layers, while word-sense disambiguation is better
predicted in the higher-level layers.

<div class="text-center mb-5">
  <img src="https://me.jiho.us/images/posts/taglm.png"
       class="img-fluid" alt="Model Architecture of TagLM"
       style="max-width:80%"/>
  <div class="caption text-center">
    Figure. Model Architecture of TagLM (image from this
    [<a href="https://arxiv.org/pdf/1705.00108.pdf">paper</a>])
  </div>
</div>

The above figure illustrates how a word in its context is tranformed into a
contextualized embeddings and how the contextualized embeddings are being
used along with the context-independent embeddings by concatenation in a
specific NLP task.

Right side of the figure is a pre-trained bi-drectional language model
(biLM). The contextualized embeddings can be constructed by computing the
weighted-average of the intermediate layer representations of the language
model. The weights are optimized to a specific task.

As formally described in the paper, each word representation is computed as
below:

$$
ELMo_k  = \gamma^{task} \sum_{j=0}^L s_j^{task} h_{k,j}^{LM}
$$

* $$h_{k,j}^{LM}$$ is the concatenated hidden outputs of each biLSTM layer,
* $$s^{task}$$ are softmax-normalized weights,
* the scalar parameter $$\gamma^{task}$ allow$s the task model to scale the entire ELMo vector.

### External Links

* [ELMo at AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)


## BERT: Bidrirectional Encode Representations from Transformers

*original paper:
[BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding](https://arxiv.org/pdf/1810.04805.pdf)*

BERT is another method of training language model, which is composed of the
Transformer encoders.

### The Transformer: 

This method is first introduced in the paper, [Attention is All You
Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf), by a
Google Brain team (Vaswani et al.) which model is for machine translation.
Traditionally, for language translation, RNNs were used for encoding
seqauences of words. In Transformer, the recurrence component is replaced with
multi-layer attention encoders (Transformers).

There exist several issues with the RNN-based language modeling architectures,
which Transformer-based models might be able to resolve.

1. RNNs are difficult to parallelize computations to utilize GPU capability.
That is, the current state of the recurrence is dependent on the previous
state.
2. RNNs are hard to maintain the previously encoded knowledge in long-range
sequence throughout the recurrence. Attention mechanisms allow modeling
dependencies without regard to their distance in the input or output
sequence.

<div class="text-center mb-5">
  <img src="https://me.jiho.us/images/posts/transformer.png"
       class="img-fluid" alt="The Transformer" style="max-height:500px;"/>
  <div class="caption text-center">
    Figure, The Transformer (image from the original paper)
  </div>
</div>

As shown in the above figure, the Transformer model also has two parts of the
conventional machine translation architecutres; the left-side is the encoder,
and the right-side is the decoder. The input/output sequences are mapped to
token representations. In an attention-based model, we do not have a
reccurrence network, hence, each token does not contain any positional
information in its sequence. Thus, we provide additional information via
positional encoding which encodes where the current token occurs in the
sequence. If you need more details on this, please refer to the original paper.

Now, the embeddings are passed through the multiple layers of the Transformer
blocks. Each block has two layers; (1) Multi-Head Attention layer and (2) feed
forward layer. Each of these layers has residual connection followed by a
normalization procedure. The fundamental idea of attention mechanism is
implemented in this Multi-Head Attention layer.

#### Multi-Head Attention

A multi-head attention layer reads in a sequence of words and computes how
relevant each token is to its given sequence. Each attention head reads the
sequence of words and interpretes them in three different aspects; *query*,
*key*, and *value*. Intuitively, the *query (Q)* represents the information we
are looking for, the *key (K)* represents the information for computing the
relevancy to other words, and the *value (V)* represents the actual value of
the input sequence.

The relevancy between tokens are calculated by the dot product between $Q$
and $K$ vectors. In fact, this should be interpreted as the similarity
between the two vectors, which is used as the attentions score between two
words. Then, the score is divided by the squre root of the dimension of the
key vector for gradient stability. Applying softmax returns the normalized
scores. Finally, the actual value $V$ of the input sequence is weighted by
the attention score. Formally, this entire procedure can be defined as below:

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

This attention function is applied multipe times on the same input sequence,
and the output results are concatednated. The following feedforward layer
projects the outputs once again and pass it to the next following transformer
block.



### Training the Language Model

BERT is trained using two different prediction tasks: (1) Masked Language
Model (MLM) and (2) Next Sentence Prediction (NSP). In MLM, a certain
percentage of the input tokens are masked at ranom. The objective is to
predict the masked tokens given the surrounding words. In NSP, an example
consists of two spans of sentences sampled from a monolingual corpus. 50% of
the examples are consecutive spans and the others are randoms sentences. The
objective of the model is to predict whether a sentence of an example is the
next following sentence of the other or not.

After the language model is trained, BERT can be adopted for an NLP
task which model adds a small number of parameters just for the task.
In the following fine-tuning procedure, all of the parameters of BERT and the
additional layers are fine-tuned jointly to maximize its objective function.

### External Links

* [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
* [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
* [BERT git repo by Google-Research](https://github.com/google-research/bert)


# References

* Y. Bengio, R. Ducharme, P. Vincent, and C. Jauvin, "A neural probabilistic language model," Journal of machine learning research, vol. 3, no. Feb, pp. 1137–1155, 2003.
  \[[pdf](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)\]
* Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space." arXiv preprint arXiv:1301.3781 (2013).
  \[[pdf](https://arxiv.org/pdf/1301.3781.pdf)\]
* Zhang, Xiang, Junbo Zhao, and Yann LeCun. "Character-level convolutional networks for text classification." Advances in neural information processing systems. 2015.
  \[[pdf](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)\]
* Bojanowski, Piotr, et al. "Enriching word vectors with subword information." Transactions of the Association for Computational Linguistics 5 (2017): 135-146. 
  \[[pdf](https://www.mitpressjournals.org/doi/pdfplus/10.1162/tacl_a_00051)\]
* Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.
  \[[pdf](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)\]
