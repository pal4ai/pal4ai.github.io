---
title: "Pre-transformer Era of NLP (Part 3) - Language Models"
permalink: /blogs/IntroNLP/PTE_part_three
author_profile: false
sidebar:
    title: Intro to NLP
    nav: intro_nlp_sidebar
excerpt: ""
toc: true
---

In the previous two articles  [Part 1](/blogs/IntroNLP/PTE_part_one) and  [Part 2](/blogs/IntroNLP/PTE_part_two), we covered how pre-transformer era NLP pipelines were built — tons of data preprocessing, and extracting and engineering features that often included calculating sparse embeddings of the input text.

In this article, we advance one step further in the evolution of NLP models. Here, we will cover the following topics:

-   What is a Language Modeling?
-   N-Gram Language Modeling
-   Neural Language Modeling
-   What is RNN? Understanding how RNNs work in language modeling tasks.

Prerequisites: Introduction to Neural networks

# What is Language Modeling?

Language modeling is a technique where we assign a probability to a piece of text. This probability is calculated over the whole vocabulary. The vocabulary can have a few thousand words/sentences/pieces of text or millions of them.

Language modeling can find its applications in many tasks.  
Some examples are:

-   **Next Word prediction**: You must have seen great recommendations for the next word when you type a query on a search engine or write your emails. Here the model assigns a probability to all the possible words in the vocabulary based on the query you have typed and gives suggestions for what the next word should be based on the top  _k_ probable words.

![](https://miro.medium.com/v2/resize:fit:700/1*VQDcVtB2FRpM-yV_e6_eKg.png)

Figure 1: Example of next word prediction — auto Complete

-   **Machine translation** — When translating from one language (French) to another (English), the model takes the French sentence and generates the English translation word by word. It predicts the probability of the next English word, based on the French input sentence and the previous English word predicted.

Language modeling is ubiquitous in any text generation task. All the current GPT family models are very sophisticated language models, but language models were not always this perfect. Earlier language models were far less intricate and less charming than their contemporary peers. However, the motivation and intuition behind them are worth learning.

# A Deeper Dive into Language Modelling

Let us first discuss how generally any language model works. As discussed above, a language model tries to predict the next word at any time step  _t._ As soon as you start typing your query on the search engine, the model takes the words you have already typed and based on that context, assigns a probability (of what could be the next word) to all the words in its vocabulary. It then predicts the highest probability words as the possible next word.

For example, for the below sentence, the model assigns the highest probabilities to word  _learning_  and  _NLP_, based on the context of the sentence. This is also called the task of next-word prediction.

![](https://miro.medium.com/v2/resize:fit:700/1*uEn-xgHTkQIbp4ftzBP2bg.png)

Figure 2

Let’s see how the math works

The probability of any word at time  _t_ is calculated using the chain rule in probability.

> **Definition: Chain Rule**
> 
> P(B|A) = P(A,B)/P(A) or P(A,B) = P(A)P(B|A)
> 
> In general Chain rule for more variables:
> 
> P(x1,x2,x3,…,xn) = P(x1)P(x2|x1)P(x3|x1,x2)…P(xn|x1,…,xn-1)

Thus, for any sentence with words w_1….w_n. The joint probability of the sentence can be calculated as:

![](https://miro.medium.com/v2/resize:fit:420/1*c8rj8YeA2wr0Vew24VDETQ.png)

But how should these probabilities be calculated? One way can be calculating the counts, i.e. the probability of the next word being ‘_learning’_ given the sentence  _‘NLP is not that hard, I like’_  is the number of times the sentence  _‘NLP is not that hard, I like learning ’_  appears across all documents in the vocabulary divided by the number of times the sentence  _‘NLP is not that hard, I like’_  appears.

![](https://miro.medium.com/v2/resize:fit:387/1*SEoNnjrzDO1OVCEpS2qsPg.png)

Can you see the problem with this approach?  
Yes! All these words need to appear together exactly in this order and the chances of this happening are not always high, hence there will be a lot of zero probability scores.

Can we do any better? Andrei Markov insists that we can

> Andrey Markov was a mathematician who came up with the Markov assumption

In simpler terms, Markov's assumption states that a future state, in our case the next word can be predicted using a relatively short history, i.e. we do not need to depend on all the previous occurring words in the sentence to predict the next word. A smaller set of the last few words should be good enough.

P(learning | NLP is not that hard, I like) ~= P(learning | I like)

OR

P(learning | NLP is not that hard, I like) ~= P(learning | hard,I like)

This leads us to our first language model — the N-gram language model.

# 1. N-Gram Language model

Instead of taking all the previous words in a sentence to predict the next word. An n-gram model only takes the previous n words to predict the next word. If n=1 it is a unigram model, here the probability of each word is calculated independently by counting its occurrence in the corpus. Unigram models do not take previous words into account.

Whereas n=2 is a bi-gram model, it will count the probability of two words occurring together. Let us understand using an example, consider below three sentences:

<s> I like Pizza </s>

<s> I like coffee </s>

<s>Pizza was cold</s>

Here <s> is the start of sentence token and </s> is the end of sentence token. These tokens help the model predict the first and last word of the sentence.

Now the probability of the word  _‘I’_ given the previous word being <s> is:  
P(I|<s>) = 2/3 because  _‘I’_  occurs after <s> in two sentences out of 3.

Similarly, P(Pizza|like) = 1/3, and P(cold|coffee)=0 because they never occur together. N-gram models are also sparse with most of the probabilities being zero because they calculate probabilities for all possible bi-gram pairs.

The performance of n-gram models becomes better as we increase n and have sufficient training data.

However, n-gram models have challenges when the sentences are long.

**Example**: It was raining cats and dogs, so Pam had to carry an umbrella.

Here to predict the word ‘umbrella’, even the n=5 gram model will not be able to generate it accurately because the main context word  _‘ raining ’_  is 10 words apart. Hence, capturing context becomes hard in these long-range dependency sentences.

In summary, the pros and cons of the N-gram model are:

Pros:

-   Fast and efficient when lots of co-occurring terms in the dataset.

Cons:

-   N-gram models can still be sparse with lots of terms being zeroes.
-   Does not take into account the context of each word i.e. it does not leverage the semantics.
-   Fails to handle long-range dependencies between words.

# 2. Neural Language Models

N-gram models had a significant shortcoming, they represented text by just counting a sequence of words in a corpus. This is very limited as it fails to take into account word synonyms and semantically similar phrases to predict the next token. For example, there is no way to know that the words  _happy_  and  _delighted_ are synonyms.

Leveraging dense embeddings solves this issue. In short, dense embeddings represent words in an n-dimensional vector space. The words closer in meanings are also closer in this vector space.

> Read more about embeddings :  [Pre-transfomer Era-NLP -Part 1](https://medium.com/pal4ai/the-pre-transformer-era-of-nlp-part-1-2103eff9744)

The neural Language Model is a neural network-based model, where we pass the input text and get its vector embeddings. These embeddings are then used to predict the next word.

Any neural network model has two phases — forward propagation and backward propagation.

> **Definition:**  
> Forward propagation — In this step, we pass the input to the model and get the output. We call this output our predictions.
> 
> Backward propagation — We evaluate the predictions we get through forward pass against the actual value (supervised learning where we have the correct labels) and compute the loss. The greater the loss the further away are our predictions from the actual values. We aim to reduce the loss and iteratively change the weights or parameters of the models unless we are no longer able to reduce the loss.
> 
> If you are unaware of these concepts — we strongly recommend to watch-  [Josh stammer’s video](https://www.youtube.com/watch?v=IN2XmBhILt4&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=5)

Let us understand the intuition behind a Neural language model in more detail:

![](https://miro.medium.com/v2/resize:fit:700/1*oFnHhG2JidUIjc5Ul2mY2w.png)

Figure 3: Neural Language Model

1.  **Neural Network Composite Function**: Let’s say we have a sentence  _‘NLP is not that hard, I like’_, and we want to predict the next word. To be able to do so, we need to capture the meaning of the sentence in a vector embedding of let’s say d-dimensions — context embedding. This is done by the composite function. Initially, each word in the sentence is converted into a dense vector embedding (using for eg. Word2Vec), and then all the words are passed through (forward pass) the composite function to build the d-dimensional context embedding. The job of the composite function is to capture the relationships between the words and the underlying semantics of the sentence. Examples of composite functions are Recurrent Neural Networks (RNN) and Transformers.
2.  **Feedforward layer**: This neural layer projects our d-dimensional vector representation into a V-dimensional space. Here V is the size of the vocabulary. Intuitively this step gives a real-valued score to each word in the vocabulary given a prefix sentence.
3.  **Softmax**: Once we have real-valued scores for each word we convert them to values between 0–1 using the softmax function. These values are probabilities of each word and the sum of the probabilities for words in vocabulary  _V_  is equal to 1. The highest probable word is the next word for the given prefix. In our case for our prefix,  _‘NLP is not that hard, I like’,_ the two most probable words are  _NLP_  and  _Learning_.

_The below step is a huge step in any Neural network training process. We strongly recommend reading more about this. Here we just explain the intuition._

**4. Loss and backpropagation**: We want our model to give a prediction that is as close to the real answer as possible. Hence, we compare the predicted next word with the actual next word in the corpus and compute the loss using a loss function (many different kinds of loss functions can be used based on the problem you’re trying to address). To minimize the loss, we then adjust the following parameters—word embeddings, neural network composite function, and feedforward layer. We then again run forward propagation with updated weights. We do this iteratively until we can’t reduce the loss any further.

# Recurrent Neural Network — RNN

We briefly talked about the composite functions above. These functions aim to capture the context of the prefix sentence effectively. RNN is a type of composite function.

RNN stands for Recurrent neural network. What does it mean? It means it has a loop.

A feedforward neural network takes some input, passes it through some hidden layers, and generates an output. Everything is unidirectional.

Whereas in a Recurrent neural network, this happens in a loop:

-   At any time  _t,_ the input vector is passed through the hidden layers and an output vector is generated.
-   At time step _t+1,_ the output vector from time step  _t_ along with the next input vector is fed to the hidden layers, and another output vector is generated. This cycle goes on till all the inputs are processed.

Hence at any time step, the output from the previous time step is passed as a part of the input to the next time step. We can see how this type of model can help in solving tasks where inputs are sequences — where the input at  _t+1_  is dependent on the input at  _t._ Examples of input sequences could be time series inputs like daily weather (the task could be to predict the next day’s temperature) stock market prices, and even a string of words (which is what we are focussing on :p). A sentence is a sequence of words and these words are related to each other to form a coherent and meaningful sentence — we just need to capture these relations accurately to be able to capture the meaning of the sentnece. RNNs have proven to be essentially helpful here.

Let us understand the working of RNNs using an example.

![](https://miro.medium.com/v2/resize:fit:700/1*Dv4RgUpYcLbn9SmDX5cIKg.png)

Figure 4: Flattened RNN

The above diagram shows a flattened version of RNN i.e. how the RNN looks across the time steps.

At any given time  _t_, the RNN takes as input a word embedding of the current time step and the output hidden state of the previous time step. This output hidden state of the previous time step provides the context of all the previous words that occurred before time  _t_.

Let us understand the above diagram step by step:

-   At time step  _t1_, the input word  _‘NLP’_  is passed into the RNN. Since this is the first word, a randomly initialized previous state h0 is passed as there is no previous word before the first word. The RNN then generates an output  _h1_  which is the representation of the input at time  _t1_  and the output from the previous state. The output of RNN at any given time is referred to as a hidden state here.
-   Similarly, at time step  _t2_, previously hidden state  _h1_  is passed and the second word  _‘is’_  is passed, and a new representation  _h2_  is created. The representation  _h2_ holds the context information of the words ‘_NLP’_  and  _‘is’._
-   This continues until we reach the last word and now want to predict the next word. Based on Figure 3, the last hidden state output  _h7_  is the context embedding. This last hidden state embedding has all the information or context of the previous words in the string which can be used to help predict the next word.

RNN uses Backpropagation through time (BPTT) to update its parameters. It is a non-trivial concept and beyond the scope of this lecture. The aim of this article is to build an intuition behind RNN as it serves as a building block to future NLP concepts. RNN is not very widely used today for NLP. If you want to learn RNN in more detail kindly check out the references at the end.

# Problems with RNN

RNNs were a great upgrade over n-gram models because of their ability to capture the context from the previously occurring words. This made predicting the next word task much better than simple counting and dividing without capturing the full previous context.

However, the ability of RNNs to capture the context of all the previous words fails when the sentences become too long. In theory, they should be able to capture the context from all the previous words in the last hidden layer (_h7_  in Figure 4) but in practice when sentences become too long, RNNs focus much more on the recently occurring words and start losing context about previous words. This is also known as the  **RNN Bottleneck issue.** To solve this issue,  **attention**  was introduced — which led to an “NLP Revolution”. We will discuss attention in our next article.

# References:

-   [NLP UMass Amherst lecture on Neural Models and RNN](https://people.cs.umass.edu/~miyyer/cs685/slides/02-neural-lms.pdf)
-   [Neural Language Models by Jurafsky](https://web.stanford.edu/~jurafsky/slp3/7.pdf)