
# Practicing NLTK

Exploring Python package NLTK 

## Authors

- [@Siri-7](https://www.github.com/Siri-7)


## Installation and Code

```bash
!pip install nltk

import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

#sentence tokenization
text = """Hello, Sirisha! Welcome to your first coding program for text analytics. Text mining is also referred to as text analytics. Text Mining processes the text itself but NLP processes the metadata. NLP is one of the components of text mining. Text analytics will give you numbers to work with your data, NLP will analyze it."""
tokenized_text=sent_tokenize(text)
print(tokenized_text)

from nltk.tokenize import word_tokenize

tokenized_word = word_tokenize(text)
print(tokenized_word)

from nltk.probability import FreqDist

fdist = FreqDist(tokenized_word) #function for figuring out the frquency distribution, there are 13 tokens in the sentence sample
print(fdist)

fdist.most_common(3)

import matplotlib.pyplot as plt

fdist.plot(30, cumulative=False) #summary of this data set showing frequency of items less than or equal to the upper class limit of each class
plt.show()

nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
print(stop_words)

filtered_sent = []

for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
        
print("Tokenized Sentence:", tokenized_word)
print("Filtered Sentence:", filtered_sent)

# Lexicon Normalization
# Stemming

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words = []
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))
    
print("Filtered sentence:" ,filtered_sent)
print("Stemmed sentence:" , stemmed_words)

nltk.download('wordnet')

# POS Tagging

sent = "Sirisha has 10 hours of graduate hourly project work to do every week"

tokens = nltk.word_tokenize(sent)
print(tokens)

nltk.download('averaged_perceptron_tagger')

nltk.pos_tag(tokens)

import numpy as np

nltk.download('twitter_samples')

from nltk.corpus import twitter_samples as ts
ts.fileids()

samples_tw = ts.strings('tweets.20150430-223406.json')

tweet_token = samples_tw[1]

tokenized_1 = word_tokenize(tweet_token[1]) #needs only single string, wont work as a list
print(tokenized_1)

# building custom tokenizers - below code will detect only alphanumeric characters

from nltk import regexp_tokenize
patn = '\w+'
regexp_tokenize(tweet_token[1], patn)


# detects only punctuations

patn = '/w+| [!,\-, @, #]'
regexp_tokenize(tweet_token[1],patn)


# Exploratory Analysis of text

import nltk
from nltk.corpus import webtext
nltk.download('webtext')
webtext_sentences = webtext.sents('firefox.txt')
webtext_words = webtext.words('firefox.txt')
len(webtext_sentences)
len(webtext_words)

fdist_web = FreqDist(webtext_words)
fdist_web



#### to be continued....


```
    