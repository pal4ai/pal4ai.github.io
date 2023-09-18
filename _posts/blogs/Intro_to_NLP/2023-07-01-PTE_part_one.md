---
title: "Pre-transformer era of NLP (Part 1) - Sparse and Dense Embeddings"
permalink: /blogs/IntroNLP/PTE_part_one
author_profile: false
sidebar:
    title: Intro to NLP
    nav: intro_nlp_sidebar
excerpt: ""
toc: true
---

In the distant past, after humans had resolved basic needs like food and safety, they turned their attention to enhancing their quality of life. This desire for improvement led to the creation of beneficial innovations like the wheel and industrial machinery.

![](https://miro.medium.com/v2/resize:fit:700/1*XkUkaT7qb3faB_HfOrGKlA.jpeg)

courtesy: Bing Image Creator

Throughout history, we humans have taken inspiration from different animals and plants for various inventions — airplanes, solar panels, and bullet trains to name a few. The one organism or in fact the most fascinating part of the organism that we have continuously tried to mimic is the human brain. First, we built calculators, then sometime later — computers, and as more time passed we’ve made great strides toward achieving the goal of creating machines that can think and understand like us (no, we’re not there yet :p).

One such facet that sets us apart from all other species found on planet Earth is the use of complex language to communicate. Building a machine that can comprehend, process, and perform an action based on just human language would bring us strides closer to achieving our goal of creating human-like or artificial intelligence. This led to the emergence of Natural Language Processing (NLP) — an area of artificial intelligence that deals with building systems or machines that can understand, process, and generate language just as any human would.

NLP can be broadly divided into two categories:

-   Natural language understanding (NLU) — helps machines understand human-like language.
-   Natural Language Generation (NLG) — helps machines generate human-like language

Recently, we have taken some giant strides toward achieving the goal of NLP in the form of ChatGPT and GPT4. But do you know how we reached here? What are transformers and why has everyone been talking about them? Well, before we dive into transformers, it’s still worthwhile to be familiar to know about their predecessors and hence we would like to give you a brief history of the evolution of NLP.

This is the first part of the article where we will introduce you to some of the ways we get computers to understand human language. The second part of the article will talk more about the machine learning and deep learning models that were used in the “pre-transformer” era.

Below is a brief timeline of the evolution of NLP:

![](https://miro.medium.com/v2/resize:fit:1000/1*l5WV5wjaksxYvB61xAMS6w.jpeg)

Some of the tasks of NLP are sentiment classification, question-answering, and summarization. But before solving a particular task it is important for computers to understand the input — words and sentences. So how do computers understand human language? Well, continue reading on….

# Embeddings

Computers don’t understand words or characters in language, they only can understand and process numbers. So, the first step in making computers understand words is to convert them into a numerical form. An embedding is a numerical representation of a word or string of words that captures its meaning. Capturing the meaning of the word can have many aspects:

-   Capturing  **_word sense —_** a computer mouse is different from an animal mouse.
-   **_Relationship between words_**  — happiness and bliss are synonyms whereas happy and sad are antonyms.
-   **_Words relatedness_**— diabetes and patient are not similar but are related

More formally, embeddings are n-dimensional vectors in an n-dimensional vector space that represents the words’ meanings. In simple terms, think of a 2-dimensional (here n=2) vector space as a simple XY coordinate system where each word is represented by two numbers or x and y coordinates — take a look at the figure below. The figure portrays the main goal of word embeddings: similar words will be closer together in the 2-dimensional vector space. “n” is usually a much higher number to be able to capture the in-depth semantics of each word.

![](https://miro.medium.com/v2/resize:fit:700/1*1Z-0qeL3N6ifNnBWXHGEhw.png)

Image courtesy:  [https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf)

Once these embeddings are generated, then they can be fed as inputs into any machine learning or deep learning model to solve different tasks (various models will be covered in the coming articles).

There are broadly two kinds of embeddings — sparse and dense.

# Sparse Embeddings

Sparse embeddings are large vectors with the size of the whole vocabulary (collection of unique words) where the majority of values are zeroes. This means that if there are  _M_  words in the whole vocabulary, then the size of each embedding is of  _M-_dimensions. If we consider the entire English language —  _M_  will be around 170,000. That means each word will be represented by 170,000 numbers! That is HUGE! This is one of the disadvantages of sparse embeddings — it takes up a lot of memory (you guessed it!).

Additionally, sparse embeddings are usually count-based embeddings that fail to take aspects of word relatedness and contextual similarity (example shown below). Bag of Words and TF-IDF are two major sparse embedding models.

# Bag of Words (BOW)

The bag of words model was among the earliest and most basic methods for producing sparse representations of input words. It represents all input text or documents as a simple bag of words, with a count for each word, without considering order or grammar. This model was primarily utilized for generating features based on word frequency.

![](https://miro.medium.com/v2/resize:fit:700/1*4e889mPO1a75uANjgL3wLA.png)

# TF-IDF

The BOW model gives the raw frequency of words but it is not the best measure of association between words. Finding an association between two words can be diluted by the presence of frequently occurring words like —  _a, an, the,_ etc_._

TF-IDF mitigates this problem. It stands for  _term frequency—inverse document frequency._  In Simpler terms, TF-IDF is a methodology that assigns weight to each term (word or string of words) based on its importance to a document in the whole corpus of documents.

Let us take an example of  _N_  = 3 documents where each document contains only one sentence:

-   D1 = “The lion ate the lamb”
-   D2 = “The lion was hungry”
-   D3 = “The lamb was alone”

TF-IDF has two components:

1.  **Term Frequency:** Measures frequency of word  _t_  in document  _d._ This step is the same as the Bag of Words.

![](https://miro.medium.com/v2/resize:fit:167/1*EOyBfCRB57nOg5HWlbsLNA.png)

The term frequency matrix will have terms from the whole vocabulary of the corpus as rows and the documents or sentences as columns. The cell values will have the frequency of each term in that sentence

![](https://miro.medium.com/v2/resize:fit:671/1*zWmI4ObWc0oN-c7XBfAwBA.png)

2.  **Inverse Document Frequency**

The Document Frequency of a term  _t_, refers to the number of documents in which it appears. Conversely, the Inverse Document Frequency is the reciprocal of this value. This concept is based on the notion that if a term appears frequently across all documents, it may not be significant or informative enough for any task. Examples of such terms may include common adjectives, articles, and other parts of speech in the language that occur very frequently. Below is the formula to calculate the IDF for  _N_  number of documents of a term  _t_

![](https://miro.medium.com/v2/resize:fit:505/1*thfmqlP4UY0Jwun3jH-nxg.png)

The table below shows the Inverse document frequency of each term. Example term ‘ate’ occurs in one document, thus the IDF = Log10(3/1). The term ‘the’ appears in all three documents thus its IDF is 0.

![](https://miro.medium.com/v2/resize:fit:268/1*VBU_BQc-Xx1rcHTsSut-vw.png)

TF-IDF for each term is

![](https://miro.medium.com/v2/resize:fit:205/1*8EFj_QT4Y5DEEDPMSGjVlg.png)

![](https://miro.medium.com/v2/resize:fit:669/1*J03nkN4M3d2L7Cr3tKHA9w.png)

TF-IDF assigns a weight to each word in accordance with the relevance of each term across the corpus. Each document/sentence is represented as a vector of the scores of the terms. Example “ The lion ate the lamb” has a vector representation [0.477,0,0,0.176,0.176,0,0].

Two similar vectors will have similar scores for each term in the corpus and hence will be closer in vector space.

One particular disadvantage of TF-IDF and sparse embeddings as an extension is that they cannot capture the semantic meaning of the sentences. For example, the below 2 sentences will have the same scores in TF-IDF or BOW models even though the sentences have completely different meanings:

-   The Lion ate the lamb
-   The lamb ate the lion

TF-IDF is still relevant in tasks where we need to find lexical (or word) overlap such as document retrieval in search engines, keyword matching, etc. It also offers more interpretability than dense embeddings and can be of relevance where interpretability is a priority.

# Dense Embeddings

Unlike long sparse vectors with most values as 0s and spanning to the size of the vocabulary, dense vectors are  _d_-dimensional integers values where  _d_  ranges from 50–1000, making them way smaller than sparse vectors. The smaller size means a lesser number of model parameters (which will be covered in the next part) to learn which helps in avoiding overfitting and helps with generalization. Dense embeddings capture the deeper aspects of word semantics and context way better.

The below diagram shows the dense vector representation of words generated by Word2Vec. These vector representations not only captured the syntactic information but also showcased how simple vector arithmetic could capture similar words. For example a vector of king — man + woman results in a vector that is very close to the representation of the word queen.

![](https://miro.medium.com/v2/resize:fit:700/1*3-TkisnsAi4vUFWsfrAGow.png)

Image courtesy  [Dense Vectors | Pinecone](https://www.pinecone.io/learn/dense-vector-embeddings-nlp/)

In the pre-transformer era, Word2Vec and GloVe were the two main methods for producing dense embeddings. These methods have a one-to-one mapping between a word and its embedding representation. Irrespective of the task — the word embeddings remain the same or “static”. Hence, the embeddings they produce are called  **static embeddings**. Transformer models like BERT produce  **dynamic contextualized embeddings**, which means that the same word can have different embeddings in different contexts. The dynamic embeddings will be covered in detail in future articles.

At that time, Word2Vec has been revolutionary as it opened many doors in the field of NLP. In brief, one of the main ideas behind Word2Vec is given below as an example:

-   Let’s take an example sentence:  _the fox jumped over the moon_
-   To generate embedding for the target word  _“jumped”_, we select two words immediately before (_the, fox)_ and after (_over, the)_
-   These four-word embeddings are passed through a model as shown in the figure below (quite similar to a neural network), and we get another embedding as an output.
-   The model then tries to classify the output embedding as the target word “_jumped”_.

![](https://miro.medium.com/v2/resize:fit:280/1*h78T6lE2Hm3S7yFUr8X_2g.jpeg)

CBOW Architecture. Image Inspired from  [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)

For a more in-depth understanding of Word2Vec, you can read:

-   [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
-   [https://en.wikipedia.org/wiki/Word2vec](https://en.wikipedia.org/wiki/Word2vec)

The next article will cover more details on some machine learning and deep learning models in the pre-transformer era.

P.S.: Hope you enjoyed this blog! Please feel free to comment and ask any doubts you may have. Thank you!