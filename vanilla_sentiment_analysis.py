from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import random
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def clean_text(tokens):
    tokens = [word.lower() for word in tokens]
    tokens = [token for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer =  WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

#getting the data from the corpus
all_tagged_movie_reviews = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

random.seed(1234)
train_tagged_movie_reviews,test_tagged_movie_reviews = train_test_split(all_tagged_movie_reviews,train_size=0.9)

#Preprocessing the data
train_tagged_movie_reviews = [(clean_text(tokens), review) for tokens, review in train_tagged_movie_reviews]
train_joined_movie_reviews = [(' '.join(tokens),review) for tokens,review in train_tagged_movie_reviews]

#unpacking the data
train_joined_sents,train_reviews = zip(*train_joined_movie_reviews)

test_tagged_movie_reviews = [(clean_text(tokens),review) for tokens,review in test_tagged_movie_reviews]
test_joined_movie_reviews,test_reviews = zip(*[(movie_review[0],movie_review[1]) for movie_review in test_tagged_movie_reviews])
test_joined_movie_reviews = [' '.join(tokens) for tokens in test_joined_movie_reviews]

#initializing the vector
vectorizer = TfidfVectorizer(max_features=50000)
X_tfidf = vectorizer.fit_transform(train_joined_sents)

#training the classifier with training data
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_tfidf, train_reviews)

#testing the model with testing data
X_test_tfidf = vectorizer.transform(test_joined_movie_reviews)
y_pred = naive_bayes_classifier.predict(X_test_tfidf)

#getting the accuracy and printing it
accuracy = accuracy_score(test_reviews, y_pred)
print(f'Accuracy: {accuracy*100}%')
print(classification_report(test_reviews, y_pred))

