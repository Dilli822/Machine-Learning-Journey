# List of positive and negative reviews
positive_reviews = [
    "Absolutely fantastic! The movie had a great storyline and superb acting.",
    "Loved every minute of it! The action scenes were thrilling and well-executed.",
    "A masterpiece. The director did an excellent job, and the cast was phenomenal.",
]

negative_reviews = [
    "Not worth my time. The plot was predictable and the acting was subpar.",
    "Terrible movie. The storyline made no sense and the characters were poorly developed.",
    "I didn't enjoy it at all. The pacing was slow and the ending was disappointing.",
]

# Sample test reviews
# positive_testreview = "I like it. Movie was awesome everything storyline,actione, climax. I loved it"

positive_testreview = "Absolutely fantastic! The movie had a great storyline and superb acting."
negative_testreview = "Not good I didnot like it!"

# Function to classify a review
def classify_review(review):
    if review in positive_reviews:
        return "Positive"
    elif review in negative_reviews:
        return "Negative"
    else:
        return "Unknown"

# Classify the test reviews
positive_result = classify_review(positive_testreview)
negative_result = classify_review(negative_testreview)

print(f"Review: '{positive_testreview}' is classified as: {positive_result}")
print(f"Review: '{negative_testreview}' is classified as: {negative_result}")
