from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.corpus import treebank
import random
import nltk
import json
import numpy as np
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from viterbi2 import viterbi, prob_calc

nltk.download('wordnet')
#Calculating the probabilities required by the viterbi function
tags, start_prob, transition_prob, emission_prob = prob_calc(
    treebank.tagged_sents(tagset='universal'))


class CustomTfidfVectorizer(TfidfVectorizer):
    def __init__(self, custom_weights=None, **kwargs):
        self.custom_weights = custom_weights
        super(CustomTfidfVectorizer, self).__init__(**kwargs)

    def fit(self, raw_documents, y=None):
        super(CustomTfidfVectorizer, self).fit(raw_documents, y)
        if self.custom_weights:
            self._idf_diag = np.log(
                (1 + self._idf_diag) / (1 + self._idf_diag - self.custom_weights))
        return self


def clean_text(tokens):
    tokens = [word.lower() for word in tokens]
    tokens = [token for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


all_tagged_movie_reviews = [(list(movie_reviews.words(fileid)), category)
                            for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

random.seed(1234)
train_tagged_movie_reviews = all_tagged_movie_reviews
train_tagged_movie_reviews = [(clean_text(tokens), review)
                              for tokens, review in train_tagged_movie_reviews]

train_sents, train_reviews = zip(*train_tagged_movie_reviews)
path_list = list(viterbi(train_sent, tags, transition_prob,
                 emission_prob, start_prob) for train_sent in train_sents)

train_pos_tagged_sents = [list(zip(train_sent, path_list))
                          for train_sent, path_list in zip(train_sents, path_list)]


# Uncomment this if you dont want to compute the probabilities again and again

# file_path = 'train_pos_tagged_sents.json'
# with open(file_path, 'w') as json_file:
#     json.dump(train_pos_tagged_sents, json_file)

# file_path = 'train_pos_tagged_sents.json'
# with open(file_path, 'r') as json_file:
#     train_pos_tagged_sents = json.load(json_file)


sentiment_weights = {
    'ADJ': 2.0,
    'ADP': 1.0,
    'ADV': 2.0,
    'CONJ': 1.0,
    'DET': 1.0,
    'NOUN': 1.3,
    'NUM': 1.0,
    'PRT': 1.0,
    'PRON': 1.1,
    'VERB': 1.5,
    'X': 0.0,
    '.': 0.0,
}

#Converting the words and pos tags in a single string so we can vectorise it later
custom_tokens = [' '.join([f'{word}{tag}' for word, tag in tagged_sentence])
                 for tagged_sentence in train_pos_tagged_sents]

#defining the custim vector
custom_vectorizer = CustomTfidfVectorizer(
    max_features=5000, custom_weights=sentiment_weights)

#Vectorizing the training data
X_tfidf = custom_vectorizer.fit_transform(custom_tokens)

#splitting the data for testing and training
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, train_reviews, test_size=0.1)

#training the classifier model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

#getting the accuracy
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

#Printing the output
print("Accuracy:", accuracy*100, '%')
print("Classification Report:")
print(report)
