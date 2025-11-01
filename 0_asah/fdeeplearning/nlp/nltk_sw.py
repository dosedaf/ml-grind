import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt_tab')

teks = 'Perekonomian Indonesia sedang dalam pertumbuhan yang membanggakan.'

tokens_kata = word_tokenize(teks)

sw_ind = set(stopwords.words('indonesian'))

kata_penting = [kata for kata in tokens_kata if kata.lower() not in sw_ind]

teks_no_sw = ' '.join(kata_penting)

print(f'teks asli : {teks}\n')
print(f'teks skrg : {teks_no_sw}\n')