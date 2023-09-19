import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import sparse
import os

stemmer = SnowballStemmer('english')

def normalize_text(text):
    """
    Apply text normalization techniques:
    - Convert text to lowercase
    - Remove punctuation
    - Handle contractions (e.g., "I'm" to "I am")
    """
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Handle common contractions
    text = text.replace("won't", "will not").replace("n't", " not").replace("'m", " am")
    return text

def tokenize(sentence):
    """
    Split sentence into an array of words/tokens.
    A token can be a word or punctuation character or number.
    """
    tokens = word_tokenize(sentence)
    return tokens

def stem(word):
    """
    Stemming = finding the root form of the word.
    """
    return stemmer.stem(word)

def preprocess_text(text):
    """
    Tokenize, stem, remove stopwords, and normalize the text.
    """
    normalized_text = normalize_text(text)
    tokens = tokenize(normalized_text)
    stemmed_tokens = [stem(token) for token in tokens]
    custom_stopwords = set(["custom", "stopwords", "list"])  # Add your custom stopwords here
    filtered_tokens = [token for token in stemmed_tokens if token not in custom_stopwords]
    return filtered_tokens

def calculate_tfidf(corpus):
    """
    Calculate TF-IDF for the given corpus.
    """
    tfidf_vectorizer = TfidfVectorizer(lowercase=False)
    corpus_text = [' '.join(tokens) for tokens in corpus]  # Convert tokenized sentences back to strings
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_text)
    return tfidf_matrix


def bag_of_words(tokenized_sentence, words):
    # Stem each word and create a bag of words
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for word in sentence_words:
        if word in words:
            bag[words.index(word)] += 1
    return bag


# Example usage with machine learning integration:

# Sample data for classification
corpus = ["This is a positive sentence.", "Another positive example.", "Yet another negative sentence.", "A negative review."]
labels = [1, 1, 0, 0]  # 1 for positive, 0 for negative

# Preprocess the text and calculate TF-IDF
corpus_tokens = [preprocess_text(sentence) for sentence in corpus]
tfidf_matrix = calculate_tfidf(corpus_tokens)

# External text data (load and preprocess from a file)
external_corpus = []

# Update the vocabulary with external data
all_words = []
for tokens in corpus_tokens:  # + external_corpus_tokens:
    all_words.extend(tokens)
all_words = sorted(set(all_words))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)

# Train a machine learning model (Multinomial Naive Bayes in this example)
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
