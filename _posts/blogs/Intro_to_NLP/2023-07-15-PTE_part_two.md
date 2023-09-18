---
title: "Pre-Transformer Era of NLP (Part 2) - Text Classification"
permalink: /blogs/IntroNLP/PTE_part_two
author_profile: false
sidebar:
    title: Intro to NLP
    nav: intro_nlp_sidebar
excerpt: ""
toc: true
---

In the  [previous article](/blogs/IntroNLP/PTE_part_one), we gave an overview of what embeddings are and their broad categories — sparse and dense. In this article, we focus on how NLP tasks can be solved using sparse embeddings (TF-IDF, Bag of Words, BM25, etc.) and classical machine learning algorithms.

**Pre-requisites**: Machine Learning Basics

To explain the flow we will take an example of a text classification task.

**Text Classification** task assigns each sample in a text dataset to one category/class out of two or more categories/classes. For example, a binary (two classes) text classification task could be to categorize movie reviews into positive or negative reviews.

Any machine learning pipeline including the NLP task we intend to build a solution for can be generalized in the flowchart below:

![](https://miro.medium.com/v2/resize:fit:1000/1*K8_Gj25561KQ093upMVAJg.png)

Machine Learning Pipeline

Let’s first walk through each of the steps given in the above flowchart in some detail.

# Dataset

The first step is to gather relevant datasets for your use case. A dataset is nothing but a set of samples.

The dataset that we will be referring to is the popular text classification dataset**:**  [IMDB Movie Reviews Dataset](https://aclanthology.org/P11-1015.pdf)  —a binary sentiment analysis dataset consisting of movie reviews.  
This dataset contains 50,000 movie reviews that have split into 25,000 train and test sets. The distribution of positive and negative labels in each train/test split is balanced.  
You can download the dataset from  [HERE](http://ai.stanford.edu/~amaas/data/sentiment/). The dataset is zipped in a  _.tar.gz_  format and has each text sample stored in a  _.txt_  file under directories  _train/pos_,  _train/neg_,  _test/pos_, and  _test/neg_.  
Let’s load this dataset and create two lists —  _train_data_  and _test_data_:

```python
!tar -xzvf aclImdb_v1.tar.gz  
  
import glob  
train_pos_files = glob.glob('aclImdb/train/pos/*.txt')  
train_neg_files = glob.glob('aclImdb/train/neg/*.txt')  
test_pos_files = glob.glob('aclImdb/test/pos/*.txt')  
test_neg_files = glob.glob('aclImdb/test/neg/*.txt')  
  
train_data = []  
test_data = []  
for file in train_pos_files:  
  f = open(file)  
  text = f.read()  
  train_data.append([text, 'positive'])  
  
for file in train_neg_files:  
  f = open(file)  
  text = f.read()  
  train_data.append([text, 'negative'])  
  
for file in test_pos_files:  
  f = open(file)  
  text = f.read()  
  test_data.append([text, 'positive'])  
  
for file in test_neg_files:  
  f = open(file)  
  text = f.read()  
  test_data.append([text, 'negative'])  
  
print(len(train_data))  
print(len(test_data))

Output:   
25000  
25000

```

# Data Pre-processing

The initial steps of preprocessing a text dataset are similar to any supervised dataset for machine learning . A lot of these steps depend on your use case but some common ones are:

-   Dealing with missing or NULL values — Imputing null values with mean / median or other metrics or altogether dropping them.
-   Label-encoding/one-hot encoding of category- Converting text columns into numbers. For our text classification use case, we will apply label encoding to our train and test data i.e. positive labels will be labeled as 1 and negative labels will be labeled as 0.

# Label Encoding  

```python
for data in train_data:  
  if data[1] == 'positive':  
    data[1] = 1  
  elif data[1] == 'negative':  
    data[1] = 0  
  
for data in test_data:  
  if data[1] == 'positive':  
    data[1] = 1  
  elif data[1] == 'negative':  
    data[1] = 0
```

After completing the general preprocessing steps, specific methods are applied to preprocess the text for each sample:

**a) Tokenization:  
**Tokenization involves splitting up the text sample into words, word/character  **_n-grams_**, or other meaningful segments, known as tokens.  
_Example:  
-_ Sentence: This Article is amazing  
- Word Tokens: [This, Article, is, amazing]

> **_Definition:_**  n_-grams  
> _n_-grams are a group of_ n  _contiguous elements in a text sample.  
> Taking the same example sentence as above —  
> _**_word 3-gram (_n=3)_:_**_  
> (<s>, <s>, This) (<s>, This, Article) (This, Article, is) (Article, is, amazing!) (is, amazing!, <e>) (amazing!, <e>, <e>)  
> _**_character 3-gram (_n=3)_:_**_(<s>, <s>, T) (<s>, T, H) (T, H, I) ….._ you get the point😜

For text classification, to apply the other data preprocessing steps on our data, we will first split up our samples into words only. For other NLP use cases, you may prefer n-grams or other tokenization methods.

We will be using the  _nltk_  library to help us do the word tokenization.

```python
import nltk  
nltk.download('punkt')  
from nltk.tokenize import word_tokenize  
  
for data in train_data:  
  data[0] = word_tokenize(data[0])  
  
for data in test_data:  
  data[0] = word_tokenize(data[0])
```

**b) Removing Stopwords:  
**Stopwords are those words that appear very frequently in any particular language and aren’t very meaningful.  
Some standard examples in English are articles and conjunctions like  _a, an, and, but, the,_ etc. You can print the stopwords using this piece of code:

```python
import nltk  
from nltk.corpus import stopwords  
   
nltk.download('stopwords')  
print(stopwords.words('english'))

Output:  
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
```

Based on your text dataset such as if it is very domain-specific, you can choose to add more words to the list of stopwords.  
The benefit of removing stopwords is to reduce the vocabulary size and, therefore, the size of each sparse embedding. Think back to BoW or TF-IDF, mentioned in the previous article. If you keep such non-meaningful words, the size of the vocabulary will be larger, and as each sparse embedding is equal to the size of the vocabulary, hence it becomes larger too.  
Let’s remove stopwords from our dataset.

```python
import nltk  
from nltk.corpus import stopwords  
  
# Removing Stopwords  
stop_words = stopwords.words('english')  
for i, data in enumerate(train_data):  
  filtered_sentence = [word_token for word_token in data[0] if not word_token.lower() in stop_words]  
  train_data[i] = [filtered_sentence, data[1]]  
  
for i, data in enumerate(test_data):  
  filtered_sentence = [word_token for word_token in data[0] if not word_token.lower() in stop_words]  
  test_data[i] = [filtered_sentence, data[1]]

**c) Removing Punctuations:  
**In the next step when we want to extract features, punctuations don’t really hold valuable information. Additionally, we want the sparse embeddings of “book” and “book’s” be the same.  
Let’s remove punctuations for our use case:

# Removing Punctuations  
import string  
for data in train_data:  
  for punctuation in string.punctuation:  
      data[0] = [word_token.replace(punctuation, '') for word_token in data[0]]  
  
for data in test_data:  
  for punctuation in string.punctuation:  
      data[0] = [word_token.replace(punctuation, '') for word_token in data[0]]
```

**c) Stemming:  
**Stemming involves reducing a word to its base or root form by removing a few characters from the end of the word.  
_Example:  
-_ Stemmer(“running”) → “run”  
We can see how stemming helps reduce the size of the vocabulary and also implicitly ensures that different forms of a root word have the same embedding (for eg. “run” and “running)  
There are certain issues with stemming as it can result in a word that does not exist in the dictionary.  
_Example:  
-_ Stemmer(“completely”) → “complet”  
Stemming works on a word-by-word basis.

Here is how you can do stemming using the  _nltk_  library:

```python
import nltk  
from nltk.stem import PorterStemmer  
  
stemmer = PorterStemmer()  
print(stemmer.stem('running'))

Output:  
run
```

For the text classification I choose to do lemmatization (see below).

**d) Lemmatization:  
**Lemmatization achieves the same goal as stemming, but additionally considers the context of that word in the dataset/corpus and then proceeds with reducing it to its base form or  _lemma_.  
Lemmatization also determines the word’s  **_Part-Of-Speech_**  by looking at the surrounding text which further helps it in deciding how to and what to reduce the word to.  
This method performs word reduction in a much more informative way as compared to Stemming and hence many times preferred over Stemming.

**_Note_:**  _Stemming and Lemmatization can be used together or only one at a time. It completely depends on YOU (the data scientist) on what works best._

> **_Definition:_** _Part-Of-Speech (POS)  
> The process of categorizing a word to its grammatical function/role based on the other words surrounding it (context!). In the English language there are 8 Parts-Of-Speech_POS-Tagging Example :  
> -_Sentence : I like reading informative articles.  
> -POS Tags: (I, Pronoun) (like, verb) (reading, verb) (informative, adjective) (articles, noun)_

Let’s go ahead and take a look at how we would apply Lemmatization on our dataset in code:

```python
from nltk.stem import WordNetLemmatizer  
lemmatizer = WordNetLemmatizer()  
nltk.download('wordnet')  
  
for data in train_data:  
    data[0] = [lemmatizer.lemmatize(word_token) for word_token in data[0]]  
  
for data in test_data:  
    data[0] = [lemmatizer.lemmatize(word_token) for word_token in data[0]]  
  
print(train_data[0][0])

Output:  
['Kids', '', 'whatever', 'age', '', 'want', 'know', 'parent', '', 'sex', 'life', '', 'grownup', 'child', 'often', 'seriously', 'baffled', 'disconcerted', 'evidence', 'aging', 'parent', 'posse', 'active', 'libido', '', 'Lastly', '', 'many', 'moviegoer', 'uncomfortable', 'watching', 'dowdy', '', 'frumpy', 'widow', 'would', 'pas', 'unnoticed', 'almost', 'anywhere', 'discover', 'aching', 'capacity', 'need', 'raw', 'passion', 'handsome', 'man', 'half', 'age', '', 'br', '', '', '', 'br', '', '', '', 'Mother', '', 'provocative', 'look', 'scarcely', 'filmed', 'reality', '', 'woman', 'nt', 'ready', 'stay', 'home', '', 'watch', '', 'telly', '', '', 'vegetate', 'husband', 'nearly', 'three', 'decade', '', 'controlling', '', 'dominating', 'chap', '', 'pack', 'massive', 'heart', 'attack', '', 'br', '', '', '', 'br', '', '', 'May', '', 'Anne', 'Reid', '', 'husband', 'two', 'child', '', 'dysfunctional', 'way', '', 'male', 'son', 'life', 'beautiful', 'wife', 'may', 'well', 'driving', 'Bankruptcy', 'Court', 'extravagant', 'commercial', 'venture', '', 'Paula', '', 'Cathryn', 'Bradshaw', '', '', 'teacher', 'aspiration', 'succeeding', 'writer', '', 's', 'attractive', '', 'pretty', '', 'seems', 'close', 'relationship', 'mum', '', 'first', '', 'br', '', '', '', 'br', '', '', 'Back', 'house', 'burying', 'husband', '', 'May', 'determines', 'stay', '', 'Rejecting', 'typical', 'widowhood', 'legacy', 'boring', 'day', 'adventure', '', 'go', 'stay', 'Paula', 'young', 'son', '', 'Paula', 's', 'boyfriend', '', 'Darren', '', 'Daniel', 'Craig', '', '', 'ruggedly', 'handsome', 'contractor', 'seems', 'taking', 'awfully', 'long', 'time', 'complete', 'addition', 'May', 's', 'son', 's', 'house', '', 'May', 'quite', 'taken', 'harddrinking', '', 'cokesniffing', 'Darren', 'whose', 'treatment', 'Paula', 'ought', 'alerted', 'May', '', 'sure', '', 'Fellow', 'Royal', 'Academy', 'Cads', '', 'br', '', '', '', 'br', '', '', 'follows', 'torrid', 'affair', 'Darren', 'besotted', 'bubblingly', 'alive', '', 'dare', 'say', 'reborn', '', '', 'widow', '', 'love', 'scene', 'graphic', 'take', 'second', 'place', 'amateur', 'artist', 'May', 's', 'pen', 'ink', 'sketch', 'tryst', 'play', 'role', 'enfolding', 'drama', '', 'debacle', '', 'take', 'pick', '', '', '', 'br', '', '', '', 'br', '', '', 'theater', 'Manhattan', 'packed', 'today', 's', 'early', 'afternoon', 'showing', 'well', 'half', 'audience', 'range', 'May', 's', 'age', '', 'shocked', 'disturbed', 'see', 'disporting', 'erotic', 'abandon', 'arm', 'much', 'younger', 'man', 'understatement', '', '', 'br', '', '', '', 'br', '', '', 'blindingly', 'honest', 'look', 'older', 'woman', 's', 'awakened', 'passion', 'decade', 'dutifully', 'obeying', 'husband', 's', 'desire', 'stay', 'home', 'raise', 'kid', '', 'also', 'mention', 'nt', 'like', 'friendswhat', 'guy', '', 'surface', 'number', 'issue', '', 'May', 's', 'dalliance', 'Darren', 'nt', 'constitute', 'incest', '', 'real', 'psychological', 'dimension', '', 'issue', '', 'mother', 'bedding', 'daughter', 's', 'lover', '', 'Paula', 'nt', 'made', 'stoutest', 'stuff', 'begin', '', 'affair', '', 'disclosed', '', 'allows', 'peeling', 'open', 'motherdaughter', 'relationship', '', 'Paula', 's', 'viewpoint', '', 'left', 'something', 'desired', '', 'Ms', 'Bradshaw', 'excellent', 'role', 'daughter', 'want', 'mother', 's', 'support', 'well', 'loveshe', 'nt', 'dealt', 'terrible', 'hand', 'life', 'nt', 'bed', 'rose', 'either', '', 'br', '', '', '', 'br', '', '', 'May', 'strong', 'resolve', 'acknowledge', 'sexuality', 'expect', '', 'indeed', 'demand', '', 'future', 'happiness', '', 'also', 'inescapably', 'vulnerable', '', 's', 'fishing', 'uncharted', 'emotional', 'water', '', 'control', 'relationship', 'Darren', 'difficult', 'issue', 'understand', '', 'much', 'le', 'resolve', '', 'sixty', '', 's', 'still', 'work', 'progress', '', 'br', '', '', '', 'br', '', '', '', 'Something', 's', 'Got', 'ta', 'Give', '', 'recently', 'showcased', 'mature', 'sexuality', 'amusingly', 'antiseptic', 'way', 'assuring', 'viewer', 'would', 'discomfited', '', 's', 'Jack', 'Nicholson', 'always', 'beautiful', 'Diane', 'Keaton', 'cavorting', 'world', 'rich', '', 'insure', 'serious', 'psychosocial', 'issue', 'explored', '', 'Keaton', 's', 'young', 'girlfriend', '', 'Amanda', 'Peet', '', 'daughter', 'Keaton', '', 'blesses', 'match', 'insures', 'audience', 'know', 'old', '', 'er', '', 'wouldbe', 'lover', 'never', 'hopped', 'sack', '', 'br', '', '', '', 'br', '', '', 'easy', '', 'Anne', 'Reid', 's', 'inspired', 'performance', 'force', 'discomfort', 'drawing', 'respect', 'others', '', 'naked', 'body', 'burst', 'sexuality', 'appears', 'absurd', 'object', 'physical', 'attraction', 'others', '', 'comment', 'audience', 'member', 'leaving', 'today', 'reflected', 'view', '', '', '', 'br', '', '', '', 'br', '', '', 'Kudos', 'director', 'Roger', 'Michell', 'tackling', 'fascinating', 'story', 'verve', 'empathy', '', 'br', '', '', '', 'br', '', '', '910', '']

```

You can see that there are no stopwords and punctuations, and Lemmatization has taken place on the words. We can now combine these tokens into a single continuous string.

This completes the data preprocessing steps. It is important to note that sometimes the best way to decide how to preprocess your data is only possible by carrying out multiple experiments and comparing results.

# Feature Extraction

## Description

In this step for each use case, we would need to extract and optionally engineer features to provide as inputs to our machine learning model.

Guess what?? In NLP, feature extraction is just converting the text samples to sparse embeddings! We perform TF-IDF or Bag-Of-Words on all of the samples in the dataset.

In addition to converting the text sample to sparse embeddings OR input features, we can “engineer” features too. Examples of engineered features could be: “containsNumericals”, “containsSymbols” etc.

Let’s look at how we would be preparing the feature matrix for Text Classification.

# Convert all the tokens into a string again  
#      a) Remove single character tokens  
#      b) Add only one space between each token  

```python
train_labels = []  
train_samples = []  
for data in train_data:  
  sentence = ""  
  for word_token in data[0]:  
    if len(word_token) > 1:  
      sentence += word_token + " "  
  sentence.strip()  
  train_samples.append(sentence)  
  train_labels.append(data[1])  
  
test_labels = []  
test_samples = []  
for data in test_data:  
  sentence = ""  
  for word_token in data[0]:  
    if len(word_token) > 1:  
      sentence += word_token + " "  
  sentence.strip()  
  test_samples.append(sentence)  
  test_labels.append(data[1])  

# Generating Feature Matrix  
# Using TfidfVectorizer from Scikit-Learn library  
from sklearn.feature_extraction.text import TfidfVectorizer  
vectorizer = TfidfVectorizer()  
  
# fit_transform() function is used on train set  
X_train = vectorizer.fit_transform(train_samples)  
  
# transform() function is used on test set, as that will use the same  
# vocabulary built on train set  
X_test = vectorizer.transform(test_samples)  
  
print(X_train.shape)  
print(X_test.shape)  
print(X_train[0].toarray())

Output:  
(25000, 90235)  
(25000, 90235)  
[[0. 0. 0. ... 0. 0. 0.]]
```

Here, we can see that  _N=_25000 and the dimensions or vocabulary size is 90,235. You can see that most of values are zero, hence the vector is a “sparse” vector.

Congratulations!!! You just extracted features that can be provided to a machine learning model for training.

_Note:_  It is important to note that TfidfVectorizer() applies L2-norm after calculating the TF-IDF scores for each document.

# Modeling and Predictions:

This is something most of you would already know. You can use any machine learning model once you have constructed the feature matrix — Decision Trees, Naive Bayes, Random Forest, SVMs, Logistic Regression, etc ….. You name it!

Ideally, you would want to iteratively perform model evaluation and selection til you get the best performance on your validation set.

For our use case we will use SVMs:

```python
from sklearn.svm import SVC  
classifier = SVC(gamma='auto')  
classifier.fit(X_train, train_labels)
```
This will take some time to run. Once the training is completed, let’s evaluate the performance of our model.

```python
# Running predictions on Test Set  
test_preds = classifier.predict(X_test)  
  
# Generating Classification Report  
from sklearn.metrics import classification_report  
print(classification_report(test_labels, test_preds, target_names=['Positive', 'Negative']))
```

![](https://miro.medium.com/v2/resize:fit:589/1*tyKQZ7FWsUdrylGtG3AA9A.png)

The Accuracy is 62% and the Macro-Avg F1-score is 56%.  
If we try different models and some more data preprocessing then we should be able to get better results.

Congratulations!!! We have built our first NLP classifier (many more to come….)!! In this article we showed the steps to build a text classifier. But these methods are not very prominent today. We have made huge progress in terms of understanding the text and it’s meaning by creating robust dense embeddings. We will discuss those methods and models in details in the upcoming articles.

In the next article we will talk about the first language model and its neural successor. These language models were the first generative models in NLP.