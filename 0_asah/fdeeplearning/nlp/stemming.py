import nltk
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["running", "runs", "runner", "ran", "easily", "fairness", "better", "best", "cats", "cacti", "geese", "rocks", "oxen"]

for word in words:
    stemmed = stemmer.stem(word)
    print(f'asli: {word} tapi palsu: {stemmer} \n')