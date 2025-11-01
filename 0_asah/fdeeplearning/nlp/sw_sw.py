from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from nltk.tokenize import word_tokenize

factory = StopWordRemoverFactory
sw_sastrawi = factory.get_stop_words(factory)

teks = 'Perekenomian Indonesia sedang dalam pertumbuhan yang membanggakan.'

tokens_kata = word_tokenize(teks)

kata_penting = [kata for kata in tokens_kata if kata.lower() not in sw_sastrawi]

teks_skrg = ' '.join(kata_penting)

print(f'teks asli : {teks}\n')
print(f'teks skrg : {teks_skrg}\n')