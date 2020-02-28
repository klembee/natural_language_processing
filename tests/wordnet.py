from nltk.corpus import wordnet

word = "active"

synonyms = set()
antonyms = set()

for syn in wordnet.synsets(word):
    for lemma in syn.lemmas():
        synonyms.add(lemma.name())
        if lemma.antonyms():
            antonyms.add(lemma.antonyms()[0].name())

print("Synonyms of {}: {}".format(word, synonyms))
print("Antonyms of {}: {}".format(word, antonyms))
