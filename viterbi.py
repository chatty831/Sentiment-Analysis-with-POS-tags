# Import necessary modules
from nltk.corpus import treebank
import random
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from viterbi2 import viterbi, prob_calc
from nltk.corpus import brown, conll2000, indian

brown_sentences = brown.tagged_sents()
conll2000_sentences = conll2000.tagged_sents()
indian_sentences = indian.tagged_sents()
treebank_sents = treebank.tagged_sents()

combined_tagged_sentences = brown_sentences + treebank_sents


# Reading the Treebank tagged sentences with universal tagset
all_tagged_sents = list(combined_tagged_sentences)

# Convert all words to lowercase and reformat the data structure
all_tagged_sents = [[(item[0].lower(),item[1]) for item in sublist]for sublist in all_tagged_sents]

random.seed(1234)

# Splitting into training and test sets
train_sents, test_sents = train_test_split(all_tagged_sents, train_size=0.9)

# Calculating all the probabilities using the self defined function in another file
tags, start_prob, transition_prob, emission_prob = prob_calc(train_sents)

# Prepare test data for evaluation
test_sents_1 = [[item[0] for item in sublist] for sublist in test_sents]
test_tags = [[item[1] for item in sublist] for sublist in test_sents]

# Evaluate accuracy
accuracy = []
for sent, tag in zip(test_sents_1, test_tags):
    path_list = viterbi(sent, tags, transition_prob, emission_prob, start_prob)
    correct = 0
    for i, path in enumerate(tag):
        if path_list[i] == tag[i]:
            correct += 1
    accuracy.append((correct / len(path_list) * 100))

# Print the average accuracy
print(sum(accuracy) / len(accuracy), '%')
