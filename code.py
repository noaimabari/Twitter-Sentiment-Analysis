import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string

## getting training data

df = pd.read_csv("0000000000002747_training_twitter_x_y_train.csv")
y = list(df["airline_sentiment"])
text = list(df["text"])

## training documents
documents = [] 
for i in range(len(y)):
    documents.append((text[i].split(" "),y[i])) 
documents[0:5] ## in the form of tuple with 1st element as a list of words and second as the sentiment, ie positive, negative or neutral


## processing the words:

## making a list of stopwords and punctuations to be removed from the tweets

stop = stopwords.words("english")
punctuations = list(string.punctuation)
stop = stop + punctuations ## to remove punctuations


## using nltk library 

from nltk.corpus import wordnet

## fucntion to get pos_tag of a word
def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
lt = WordNetLemmatizer() 

## fucntion to clean the words in the text 
## removing stop words and punctuations 
## performing lemmatization to get better insight on the meaning of the words

def clean_review(words):
    output_words = []
    for w in words:
        if w.lower() not in stop:
            try:
                pos = pos_tag([w])
                clean_word = lt.lemmatize(w,pos = get_simple_pos(pos[0][1]))
                output_words.append(clean_word.lower())
            except:
                continue
    return output_words

documents = [(clean_review(i),category) for i,category in documents]


y_train = [category for words,category in documents]
x_train = [" ".join(words) for words, categories in documents]


## using tfidf Vectorizer to remove words which do not hold much importance during classification

from sklearn.feature_extraction.text import TfidfVectorizer
token_vec = TfidfVectorizer(max_features = 3000, ngram_range = (1,3))
x_train_t = token_vec.fit_transform(x_train) ## we can use the sparse matrix directyly as training and testing data in our sklearn classifiers


## forming x_test

df = pd.read_csv("0000000000002747_test_twitter_x_test.csv")
text = list(df["text"])
x_test = []
for i in range(len(text)):
    x_test.append(text[i])
x_test

x_test_t = token_vec.transform(x_test) ## we can use the sparse matrix directyly as training and testing data in our sklearn classifiers


## both training and testing data now ready
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train_t, y_train)
## classification using MultiNomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train, y_train)
print(clf.score(x_val, y_val)) ## checking the score on the validation data

## saving the predictions in a csv file 
y_pred = clf.predict(x_test_t)
predictions = np.array(y_pred)
pd.DataFrame(predictions).to_csv("finall.csv") ## saving file

