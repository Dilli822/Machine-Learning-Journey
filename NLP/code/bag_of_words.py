def bag_of_words(text):
    words = text.lower().split(" ") 
    bag = {}  # stores all of the encodings and their frequency 
    vocab = {}  # stores word to encoding mapping
    word_encoding = 1  # starting encoding
    
    for word in words:
        if word in vocab:
            encoding = vocab[word]
        else:
            vocab[word] = word_encoding
            encoding = word_encoding
            word_encoding += 1
        if encoding in bag:
            bag[encoding] += 1
        else:
            bag[encoding] = 1
    
    return bag, vocab

text = "Hello. I am Dilli Hang Rai. AI enthusiast from Nepal. I am doing my bachelors in Computer Science"
bag, vocab = bag_of_words(text)
print("Bag of words:", bag)
print("Vocabulary:", vocab)

# Output
# Bag of words: {1: 1, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 
# 14: 1, 15: 1, 16: 1}
# Vocabulary: {'hello.': 1, 'i': 2, 'am': 3, 'dilli': 4, 'hang': 5, 'rai.': 6, 'ai': 7, 
# 'enthusiast': 8, 'from': 9, 'nepal.': 10, 'doing': 11, 'my': 12, 'bachelors': 13, 'in': 14, 
# 'computer': 15, 'science': 16}

# Algorithm 
# Input: Get the text you want to analyze.

# Preprocessing: Convert text to lowercase and split it into words.

# Initialization: Set up an empty bag of words and an empty vocabulary.

# Loop through words:

# For each word:
# If the word is new, assign it a unique ID and add it to the vocabulary.
# Update the bag of words by counting the occurrences of each word ID.
# Output: Return the bag of words and the vocabulary.

# End.

