import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer

lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('punkt_tab')

df = pd.read_csv('spotify_reviews.csv')

def clean(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # menghapus mention
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # menghapus hashtag
    text = re.sub(r'RT[\s]', '', text) # menghapus RT
    text = re.sub(r"http\S+", '', text) # menghapus link
    text = re.sub(r'[0-9]+', '', text) # menghapus angka
    text = re.sub(r'[^\w\s]', '', text) # menghapus karakter selain huruf dan angka
 
    text = text.replace('\n', ' ') # mengganti baris baru dengan spasi
    text = text.translate(str.maketrans('', '', string.punctuation)) # menghapus semua tanda baca
    return text

def case_folding(teks):
    return str(teks).lower()

def hapus_angka(teks):
    return ''.join([char for char in teks if not str(char).isdigit()])

def hapus_tanda_baca(teks):
    punctuation_set = set(string.punctuation)
    
    return ''.join([char for char in teks if char not in punctuation_set])

def hapus_whitespace(teks):
    return teks.strip()

def hapus_stopwords(teks):
    tokenized_words = word_tokenize(teks)
    en_stopwords = set(stopwords.words('indonesian'))
    important_words = [kata for kata in tokenized_words if kata not in en_stopwords] # kata already lower
    
    return ' '.join(important_words)

def stem(teks):
    words = word_tokenize(teks)
    stemmer  = PorterStemmer()
    stemmed_words = [stemmer.stem(kata) for kata in words]

    return ' '.join(stemmed_words)

def lemmatize(teks):
    words = word_tokenize(teks)
    lemmatized_words = [lemmatizer.lemmatize(kata, pos=wordnet.VERB) for kata in words]
    
    return ' '.join(lemmatized_words)

df = df.dropna()
df = df.drop_duplicates()

df['text'] = df['text'].apply(clean)
df['text'] = df['text'].apply(case_folding)
df['text'] = df['text'].apply(hapus_angka)
df['text'] = df['text'].apply(hapus_tanda_baca)
df['text'] = df['text'].apply(hapus_whitespace)
df['text'] = df['text'].apply(hapus_stopwords)
# df['text'] = df['text'].apply(stem)
df['text'] = df['text'].apply(lemmatize)

print(type(df['text']))


df.to_csv('preprocessed_spotify_reviews.csv', index=False)
