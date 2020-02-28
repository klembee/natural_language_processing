from nltk.stem import PorterStemmer
from nltk import word_tokenize, pos_tag
from collections import defaultdict

# Stemming
print("Stemming:")

text = "Sushis comes in multiple forms." + \
       " There is the nigiri, the maki and " + \
       " the most delicious of all: the sashimi." + \
       " The multiplicity of this food is formidable. Deliciousness guaranteed"
tokens = word_tokenize(text)

ps = PorterStemmer()
for word in tokens:
    rootWord = ps.stem(word)
    print("Root of {} is: {}".format(word, rootWord))

# Lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

print("\nLemmatization:")

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

text2 = "Programming a program that programs a programmer. Learning"
tokens2 = word_tokenize(text2)

lemmaFunc = WordNetLemmatizer()
for word, tag in pos_tag(tokens2):
    print("Lemma of {} is {}".format(word, lemmaFunc.lemmatize(word, tag_map[tag[0]])))


