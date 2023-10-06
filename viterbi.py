# Import necessary modules
from nltk.corpus import treebank, brown
from nltk.tokenize import word_tokenize
import json
import random
from sklearn.model_selection import train_test_split
from viterbi2 import viterbi, prob_calc


treebank_sents = treebank.tagged_sents(tagset='universal')
combined_tagged_sentences = treebank_sents + brown.tagged_sents(tagset='universal')

# Reading the Treebank tagged sentences with universal tagset
all_tagged_sents = list(combined_tagged_sentences)

# Convert all words to lowercase and reformat the data structure
all_tagged_sents = [[(item[0].lower(),item[1]) for item in sublist]for sublist in all_tagged_sents]

# Splitting into training and test sets
train_sents, test_sents = train_test_split(all_tagged_sents, train_size=0.9,random_state=random.randint(0,100))


# # Calculating all the probabilities using the self defined function in another file
# tags, start_prob, transition_prob, emission_prob = prob_calc(train_sents)

# Comment this if you want to calculate the probabilities again and uncomment the line above.
with open('transition_prob.json','r') as json_file:
    transition_prob = json.load(json_file)
    
with open('emission_prob.json','r') as json_file:
    emission_prob = json.load(json_file)

with open('start_prob.json','r') as json_file:
    start_prob = json.load(json_file)
    
tags = set(transition_prob.keys())
# Comment till here (if you want to)
    
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
