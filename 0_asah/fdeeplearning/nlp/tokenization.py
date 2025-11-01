from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

text = 'u a bitch u kno that but u always be coping thinking u the king or sum shit. u not bru and u chopped as hell'

tknzd = word_tokenize(text)
print(tknzd)

text2 = 'they be sayin fuck me but ion care ima js play brawlhalla and be stressed'


text = 'u a bitch u kno that but u always be coping thinking u the king or sum shit. u not bru and u chopped as hell'

tknzd = word_tokenize(text)
print(tknzd)

text = 'they be sayin fuk me but ion care ima js play brawlhalla and be stressed. also some smash. and some apex when im in the mood. ts js separate strings using delimiter bruh. python mfs be using fancy words like tokenizing. bih all u did was js walking thru letters n freak out if its the delim then separate it lmao yall dumb as hell'
tknzd = sent_tokenize(text)
print(tknzd)

