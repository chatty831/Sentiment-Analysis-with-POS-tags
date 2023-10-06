# Define the Viterbi algorithm for POS tagging
def viterbi(sentence, tags, transition_prob, emission_prob, start_prob):
    V = [{}]  # Initialize the Viterbi matrix
    path = {}  # Store the best path
    
    # Initialization step: calculate probabilities for the first word
    for tag in tags:
        if sentence[0] not in emission_prob[tag]:
            emission_prob[tag][sentence[0]] = 1
        V[0][tag] = start_prob[tag] * emission_prob[tag][sentence[0]]
        path[tag] = tag
    
    # Base case for sentences with only one word
    if len(sentence) == 1:
        prob, tag = max((V[len(sentence) - 1][tag], tag) for tag in tags)
        return (path[tag], prob)
    
    # Recursion step: fill in the Viterbi matrix for the rest of the sentence
    for t in range(1, len(sentence)):
        V.append({tag: 0 for tag in tags})
        newpath = {}
        
        for tag in tags:
            if sentence[t] not in emission_prob[tag]:
                emission_prob[tag][sentence[t]] = 1
            # Find the best previous tag and its probability
            (prob1, state) = max((V[t-1][prev] * transition_prob[prev][tag] * emission_prob[tag][sentence[t]], prev) for prev in tags)
            V[t][tag] = prob1  # Update the probability in the Viterbi matrix
            newpath[tag] = path[state] + " " + tag  # Update the best path
        
        # Normalize probabilities to prevent underflow
        for tag in tags:
            if all(value < 1e-5 or value == 0 for value in V[t].values()):
                V[t] = {k: v * 1000 for k, v in V[t].items()}

        path = newpath  # Update the best path
    
    # Termination step: find the best tag for the last word
    prob, tag = max((V[len(sentence) - 1][tag], tag) for tag in tags)
    path_list = path[tag].split()  # Convert the best path to a list of tags
    return path_list

# Calculate probabilities for the HMM model
def prob_calc(train_sents):
    words = set(item[0] for sublist in train_sents for item in sublist)  # Get unique words in the training data
    tags = set(item[1] for sublist in train_sents for item in sublist)   # Get unique tags in the training data
    
    # Initialize emission probabilities with Laplace smoothing
    emission_prob = {tag: {word: 1 for word in words} for tag in tags}
    emis_total = {tag: 0 for tag in tags}
    
    # Calculate emission probabilities
    for train_sent in train_sents:
        for word, tag in train_sent:
            emis_total[tag] += 1
            emission_prob[tag][word] += 1

    # Normalize emission probabilities
    for tag in tags:
        for word in words:
            if emis_total[tag] != 0:
                emission_prob[tag][word] = emission_prob[tag][word] / emis_total[tag]

    # Initialize transition probabilities with Laplace smoothing
    transition_prob = {tag: {tag: 0 for tag in tags} for tag in tags}
    trans_total = {tag: 0 for tag in tags}
    
    # Calculate transition probabilities
    for train_sent in train_sents:
        for i in range(0, len(train_sent) - 1):
            transition_prob[train_sent[i][1]][train_sent[i + 1][1]] += 1
            trans_total[train_sent[i][1]] += 1

    # Normalize transition probabilities
    for tag in tags:
        for tag1 in tags:
            if trans_total[tag] != 0:
                transition_prob[tag][tag1] = transition_prob[tag][tag1] / trans_total[tag]

    # Initialize start probabilities
    start_prob = {tag: 0 for tag in tags}
    
    # Calculate start probabilities
    for train_sent in train_sents:
        for tup in train_sent:
            start_prob[tup[1]] += 1

    # Normalize start probabilities
    start_prob_total = sum(start_prob.values())
    for tag in tags:
        start_prob[tag] = start_prob[tag] / start_prob_total

    return tags, start_prob, transition_prob, emission_prob