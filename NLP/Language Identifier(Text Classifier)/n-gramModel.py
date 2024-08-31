import re
from collections import defaultdict
import math
import matplotlib.pyplot as plt

def create_ngram_model(text, n):
    # Preprocess the text
    text = re.sub(r'\s+', ' ', text.lower())
    
    # Create n-grams
    ngrams = defaultdict(lambda: defaultdict(int))
    for i in range(len(text) - n + 1):
        gram = text[i:i+n]
        next_char = text[i+n] if i+n < len(text) else " "
        ngrams[gram][next_char] += 1
    
    # Convert counts to probabilities
    for gram in ngrams:
        total = sum(ngrams[gram].values())
        for char in ngrams[gram]:
            ngrams[gram][char] = ngrams[gram][char] / total
    
    return ngrams

def calculate_probability(text, model, n):
    text = re.sub(r'\s+', ' ', text.lower())
    probability = 0
    for i in range(len(text) - n):
        gram = text[i:i+n]
        next_char = text[i+n]
        if gram in model and next_char in model[gram]:
            probability += math.log(model[gram][next_char])
        else:
            probability += math.log(1e-10)  # Small probability for unseen n-grams
    return probability

# Example usage
english_text = """
The quick brown fox jumps over the lazy dog. 
Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.
"""

german_text = """
Der schnelle braune Fuchs springt über den faulen Hund. 
Natürliche Sprachverarbeitung ist ein Teilgebiet der Linguistik, Informatik und künstlichen Intelligenz.
"""

# Create trigram models
english_model = create_ngram_model(english_text, 3)
german_model = create_ngram_model(german_text, 3)

# Test sentence
# test_sentence = "witaj świecie, nie jestem Niemcem"
# test_sentence = "Hallo Welt, ich bin kein Deutscher"
test_sentence = "cats and dogs"

# Calculate probabilities
english_prob = calculate_probability(test_sentence, english_model, 3)
german_prob = calculate_probability(test_sentence, german_model, 3)

print(f"Probability of being English: {english_prob}")
print(f"Probability of being German: {german_prob}")

if english_prob > german_prob:
    print("The text is more likely to be English")
elif english_prob == german_prob:
    print("Could Not Identify the language type! Please Provide More Text on German or English Language!! ")
else:
    print("The text is more likely to be German")
    
######## BELOW THIS CODE IS FOR THE VISUALIZATION PART ONLY #####################

import re
from collections import defaultdict
import math
import matplotlib.pyplot as plt

def create_ngram_model(text, n):
    # Preprocess the text
    text = re.sub(r'\s+', ' ', text.lower())
    
    # Create n-grams
    ngrams = defaultdict(lambda: defaultdict(int))
    for i in range(len(text) - n + 1):
        gram = text[i:i+n]
        next_char = text[i+n] if i+n < len(text) else " "
        ngrams[gram][next_char] += 1
    
    # Convert counts to probabilities
    for gram in ngrams:
        total = sum(ngrams[gram].values())
        for char in ngrams[gram]:
            ngrams[gram][char] = ngrams[gram][char] / total
    
    return ngrams

def calculate_probabilities(text, model, n):
    text = re.sub(r'\s+', ' ', text.lower())
    probabilities = []
    for i in range(len(text) - n):
        gram = text[i:i+n]
        next_char = text[i+n]
        if gram in model and next_char in model[gram]:
            probabilities.append(math.log(model[gram][next_char]))
        else:
            probabilities.append(math.log(1e-10))  # Small probability for unseen n-grams
    return probabilities

def predict_language(test_sentences, english_model, german_model, n):
    predictions = []
    
    for test_sentence in test_sentences:
        english_probs = calculate_probabilities(test_sentence, english_model, n)
        german_probs = calculate_probabilities(test_sentence, german_model, n)

        english_prob = sum(english_probs)
        german_prob = sum(german_probs)

        if english_prob > german_prob:
            predicted_language = "English"
        elif english_prob == german_prob:
            predicted_language = "Undetermined"
        else:
            predicted_language = "German"
        
        predictions.append((test_sentence, predicted_language))
    
    return predictions

# Example usage
english_text = """
The quick brown fox jumps over the lazy dog. 
Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.
"""

german_text = """
Der schnelle braune Fuchs springt über den faulen Hund. 
Natürliche Sprachverarbeitung ist ein Teilgebiet der Linguistik, Informatik und künstlichen Intelligenz.
"""

# Create trigram models
english_model = create_ngram_model(english_text, 3)
german_model = create_ngram_model(german_text, 3)

# Array of test sentences
test_sentences = [
    "cats and dogs",  # English
    "Hallo Welt, ich bin kein Deutscher",  # German
    # "こんにちは世界"  # Japanese (should be undetermined)
    "sdsfsfsfsgsfgsfgsdfsrfwr"
]

# Predict languages
predictions = predict_language(test_sentences, english_model, german_model, 3)

# Visualization
colors = {'English': 'green', 'German': 'red', 'Undetermined': 'blue'}
plt.figure(figsize=(10, 5))

for i, (sentence, prediction) in enumerate(predictions):
    plt.bar([i], [1], color=colors[prediction])

plt.xticks(range(len(test_sentences)), [f'"{s}"' for s, _ in predictions], rotation=0, ha="center", fontname="Arial Unicode MS")
plt.xlabel('Test Sentences')
plt.ylabel('Prediction Confidence')
plt.title('Language Prediction for Test Sentences')
plt.legend([plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in colors], colors.keys(), loc='upper right')
plt.show()