import numpy
import nltk
from nltk.corpus import brown
from tests.glove.Dictionary import Dictionary
import math

context_size = 2
wordVectorDimension = 50
dictionary = Dictionary()


def getContextWords(sentence, position):
    if not type(sentence) is list:
        words = sentence.split()
    else:
        words = sentence

    context_words = []

    for i in range((2 * context_size) + 1):
        context_word_pos = position + (i - context_size)
        if context_word_pos == position:
            continue

        distance = abs(position - context_word_pos)

        if context_word_pos < 0:
            context_words.append(("<SOS>", distance))
        elif context_word_pos >= len(words):
            context_words.append(("<EOS>", distance))
        else:
            context_words.append((words[context_word_pos], distance))

    return context_words


def createWordToContextList():
    """
    Create a list of tuples containing the following data:
    example phrase: the dog runs over the long grass
    [
        0: ([(dog, 1), (runs, 2)], the),
        1: ([(the, 1), (runs, 1), (over, 2)], dog)
    ]

    :return: The array containing the context words with the distance to the word
    """

    word2context_words = []

    for sentence in brown.sents():
        dictionary.addSentence(sentence)

        for index, word in enumerate(sentence):
            word2context_words.append((word, getContextWords(sentence, index)))

    return word2context_words


def main():
    word2context = createWordToContextList()

    contexts_embeding_mat = numpy.random.rand(2 * context_size, dictionary.nbWords, 50)
    word_embeding_mat = numpy.random.rand(dictionary.nbWords, 50)
    hidden_to_output = numpy.random.rand(50, dictionary.nbWords)

    # Try to get better values for the embeding matrix
    for word_and_context in word2context:
        word = word_and_context[0]
        contexts = word_and_context[1]

        one_hot_word = numpy.array(dictionary.getOneHot(word))
        one_hot_contexts = \
            numpy.array([dictionary.getOneHot(context[0]) for context in contexts]).reshape((2 * context_size, dictionary.nbWords, 1))

        # embeded_word = word_embeding_mat[numpy.argmax(one_hot_word)]
        embeded_contexes = numpy.array([word_embeding_mat[numpy.argmax(one_hot_context)] for one_hot_context in one_hot_contexts])

        # Concatenate the embeded words together
        layer1 = numpy.ones((1, wordVectorDimension))

        print(layer1.shape, embeded_contexes.shape)
        for embeded_contex in embeded_contexes:
            layer1 = numpy.dot(layer1, embeded_contex)

        prediction = numpy.tanh(layer1.dot(hidden_to_output)).T

        # Compute the MSE loss
        loss = numpy.sum(numpy.exp2(numpy.subtract(one_hot_word, prediction)))

        gradient = numpy.multiply(layer1, -1)
        gradient = numpy.add(numpy.multiply(layer1, numpy.exp2(numpy.tanh(numpy.dot(hidden_to_output.T, layer1)))))

        print(gradient)

    print(dictionary.nbWords)
    print(dictionary.sentenceToIndices("The dog runs over the long grass"))


if __name__ == "__main__":
    main()
