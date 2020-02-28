from nltk.tokenize import word_tokenize
from nltk import bigrams, trigrams

text = "She reached her goal, exhausted. Even more chilling to her was that the euphoria that she thought she'd feel " \
       "upon reaching it wasn't there. Something wasn't right. Was this the only feeling she'd have for over five " \
       "years of hard work? "

tokens = word_tokenize(text)
outputBi = list(bigrams(tokens))
outputTri = list(trigrams(tokens))

print("Bigrams: {}".format(outputBi))
print("Trigrams: {}".format(outputTri))
