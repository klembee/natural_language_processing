import numpy

SOS = 0
EOS = 1


class Dictionary:
    def __init__(self):
        self.word2Index = {"<SOS>": SOS, "<EOS>": EOS}
        self.index2Word = [SOS, EOS]
        self.nbWords = 2

    def addWord(self, word):
        if not word in self.word2Index.keys():
            self.word2Index[word] = len(self.word2Index.keys())
            self.index2Word.append(word)
            self.nbWords += 1

    def addSentence(self, sent):
        if not type(sent) is list:
            sent = sent.split()

        for word in sent:
            self.addWord(word)

    def getOneHot(self, word):
        one_hot = numpy.zeros((self.nbWords, 1))
        one_hot[self.word2Index[word]] = 1
        return one_hot

    def sentenceToIndices(self, sent):
        if not type(sent) is list:
            sent = sent.split(' ')

        list_of_indices = [SOS]
        for word in sent:
            list_of_indices.append(self.word2Index[word])

        list_of_indices.append(EOS)

        return list_of_indices
