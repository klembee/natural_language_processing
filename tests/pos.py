import nltk

text = "Japanese is such a great country. It has suchis, castles and gardens !".split()
print("Text: {}".format(text))
tokens_tag = nltk.pos_tag(text)
print("Tokens: {}".format(tokens_tag))

patterns = """superimportant:{<NN.?>*<VBD.?>*<JJ.?>*<CC>?}"""
chunker = nltk.RegexpParser(patterns)
print("Chunks: {}".format(chunker.parse(tokens_tag)))

# Graphing the chunks

print("\nGraphing chunks")

text = "Rice vinegar recipe: ¼ cup seasoned rice vinegar, 1 small garlic clove and 1 cup of rice !"
tokens = nltk.word_tokenize(text)
print("Text: {}".format(text))
print("Tokens: {}".format(tokens))

tag = nltk.pos_tag(tokens)
print("Tags: {}".format(tag))

grammar = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(tag)
print("Result: {}".format(result))
result.draw()

# Couting pos tags
print("\nCounting pos tags")

from collections import Counter
text = "I am learning natural language processing using python."
text = text.lower()
tokens = nltk.word_tokenize(text)
tags = nltk.pos_tag(tokens)
counts = Counter(tag for word,tag in tags)
print("Counts: {}".format(counts))



