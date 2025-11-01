import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

words = ["Run", "Cat", "Good", "Goose", "Rock", "City", "Big", "Happy", "Run", "Sleep"]

for word in words:
    lemmed = lemmatizer.lemmatize(word.lower())
    print(f'asli : {word}, palsu : {lemmed}\n')