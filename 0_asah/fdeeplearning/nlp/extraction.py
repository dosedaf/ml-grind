from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

text_data = [
    'Saya suka makan bakso',
    'Bakso enak dan lezat',
    'Makanan favorit saya adalah nasi goreng',
    'Nasi goreng pedas adalah makanan favorit saya',
    'Saya suka makanan manis seperti es krim',
]

tokenized_data = [word_tokenize(sentence.lower()) for sentence in text_data]

model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)

word_vectors = model.wv
 
similar_words = word_vectors.most_similar('bakso', topn=3)
print("Kata-kata yang mirip dengan 'bakso':", similar_words)
 
vector = word_vectors['bakso']
print("Vektor untuk 'bakso':", vector)