from sklearn.feature_extraction.text import CountVectorizer

# Latent semantic analysis
vectorizer = CountVectorizer()
corpus = ["Then came the night of the first falling star. It was seen early in the morning, rushing over Winchester " \
       "eastward, a line of flame high in the atmosphere. Hundreds must have seen it and taken it for an ordinary " \
       "falling star. It seemed that it fell to earth about one hundred miles east of him."]

vocabulary = vectorizer.fit(corpus)
x = vectorizer.transform(corpus)

print(x.toarray())
print(vocabulary.get_feature_names())

# What is wrong with this method:
# Ignores the order and context of the words

# Word2vec

import gensim
from nltk.corpus import abc

# model = gensim.models.Word2Vec(abc.sents())
# print(model.wv.most_similar('food'))

# chat bot example
print("\nChat bot example")
import json
import pandas as pd
from nltk.corpus import stopwords
from textblob import Word
import os.path
from gensim.models import Word2Vec

model = Word2Vec()

if not os.path.isfile('word2vec/model.bin'):
    json_file = 'word2vec/intents.json'
    with open(json_file, 'r') as f:
        data = json.load(f)
        df = pd.DataFrame(data)

        # Clean the data
        print("\nCleaning the data")
        stop = stopwords.words('english')
        punctuations = ['.', ',', '!', '?', ";", ':']
        df['patterns'] = df['patterns'].apply(', '.join)

        # Everything to lowercase
        df['patterns'] = df['patterns'].apply(lambda x: ' '.join(x.lower() for x in x.split()))

        # Remove punctuation
        df['patterns'] = df['patterns'].apply(lambda x: ''.join(x for x in list(x) if x not in punctuations))

        df['patterns'] = df['patterns'].str.replace('[^\w\s]', '')

        # Remove digits
        df['patterns'] = df['patterns'].apply(lambda x: ''.join(x for x in list(x) if not x.isdigit()))

        # Remove stop words
        df['patterns'] = df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if not x in stop))

        # Lemmatize the words
        df['patterns'] = df['patterns'].apply(lambda x: ' '.join([Word(word).lemmatize() for word in x.split()]))

        print(df['patterns'])

        # Building the model
        print("\nBuilding the model")

        listOfPatterns = []
        for pattern in df['patterns']:
            li = list(pattern.split())
            listOfPatterns.append(li)

        model = Word2Vec(listOfPatterns, min_count=1, size=300, workers=8)
        model.save("word2vec/model.bin")
else:
    # Load the model
    model = Word2Vec.load('word2vec/model.bin')

# Find similar words
similar_words = model.wv.most_similar('thanks')
print("Words similar to thanks:")
print(similar_words)

similarity_multiple_words = model.wv.similarity('please', 'see')
print("Similarity between please and see: {}".format(similarity_multiple_words))

similarity_multiple_words2 = model.wv.similarity('helping', 'thanks')
print("Similarity between helping and thanks: {}".format(similarity_multiple_words2))








