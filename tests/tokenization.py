from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# Word tokenization
print("Word tokenization")
text = "Hello, my name is clement and I am a programmer."
# text2 = "今日わクレモンです. プログラマーです."
filterdText = word_tokenize(text)
# filteredTextJap = word_tokenize(text2, 'japanese')
# print(filteredTextJap)

print(filterdText)

# Sentence tokenization
print("\nSentence tokenization")

text2 = "Hello, my name is clement. I am a programmer. Who are you?"
phraseTokens = sent_tokenize(text2)
print(phraseTokens)
for token in phraseTokens:
    print(word_tokenize(token))

