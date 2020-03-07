import numpy
import nltk
from nltk.corpus import brown
from tests.glove.Dictionary import Dictionary

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
            context_words.append(("<ESO>", distance))
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
            word_index = dictionary.word2Index[word]
            word2context_words.append((word_index, getContextWords(sentence, index)))

    return word2context_words


def main():
    createWordToContextList()

    embeding_mat = numpy.random.rand(dictionary.nbWords, 50)
    print(embeding_mat)

    print(dictionary.nbWords)
    print(dictionary.sentenceToIndices("The dog runs over the long grass"))


if __name__ == "__main__":
    main()
