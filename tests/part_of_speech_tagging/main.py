import numpy as np
from torch.nn import MSELoss
from torch.optim import SGD
import torch
import os
from nltk.corpus import brown

from tests.part_of_speech_tagging.datasets import loadTestData, loadTrainData
from tests.part_of_speech_tagging.networks import POSTagger, POSTaggerReccurent


class Dictionary:

    def __init__(self):
        self.words = []
        self.word2Index = {}
        self.nb_words = 0

    def addWord(self, word):
        if not word in self.word2Index.keys():
            self.word2Index[word] = len(self.word2Index.keys())
            self.words.append(word)
            self.nb_words += 1

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word.lower())

    def getOneHot(self, word):
        one_hot = np.zeros(self.nb_words)
        if word in self.word2Index.keys():
            one_hot[self.word2Index[word]] = 1
        return one_hot

    def fromOneHot(self, one_hot):
        index = torch.argmax(one_hot)
        return self.words[index]


learning_rate = 10e-3
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")


def trainLR(network, loss_func, optimizer, x, y):
    total_loss = 0

    for i in range(len(x)):
        word = torch.Tensor(x[i]).to(device)
        target = torch.Tensor(y[i]).to(device)

        optimizer.zero_grad()

        prediction = network(word)
        loss = loss_func(prediction, target)
        total_loss += loss

        loss.backward()
        optimizer.step()

        if i % network.printEvery == 0:
            print("{}/{}: {}".format(i, len(x), total_loss / network.printEvery))
            total_loss = 0

    torch.save(network.state_dict(), "model.pt")

def testLR(network, x, y):
    """
    Test the network with the provided inputs and outputs
    :param x: the inputs to give to the network
    :param y: the expected outputs of the network
    """

    nb_rights = 0
    hidden = None

    for i in range(len(x)):
        input = torch.Tensor(x[i]).to(device)
        output = torch.Tensor(y[i]).to(device)
        prediction = network(input)

        pred_highest_index = torch.argmax(prediction)
        output_highest_index = torch.argmax(output)

        if pred_highest_index == output_highest_index:
            nb_rights += 1

        if (i + 1) % network.printEvery == 0:
            print("{} %".format((nb_rights / i) * 100))



def main():
    wordDic = Dictionary()
    posDic = Dictionary()
    train_data = loadTrainData()
    test_data = loadTestData()

    print("Creating dictionaries")
    for sentence in brown.sents():
        wordDic.addSentence(sentence)

    for pos in train_data['pos']:
        posDic.addWord(pos)

    nb_words = wordDic.nb_words
    nb_pos_tags = train_data['pos'].nunique()

    network = POSTagger(nb_words, nb_pos_tags, 256, device)

    if not os.path.isfile("model.pt"):
        # Converts the tra
        print("Converting training data to one hot")
        inputs = [wordDic.getOneHot(word) for word in train_data['word']]
        outputs = [posDic.getOneHot(pos) for pos in train_data['pos']]

        print("Training")
        loss_func = MSELoss()
        optimizer = SGD(network.parameters(), lr=learning_rate)

        trainLR(network, loss_func, optimizer, inputs, outputs)
    else:
        print("Loaded from model saved file")
        network.load_state_dict(torch.load("model.pt"))

    # Testing the network
    inputs = [wordDic.getOneHot(word) for word in test_data['word']]
    outputs = [posDic.getOneHot(pos) for pos in test_data['pos']]
    testLR(network, inputs, outputs)

    # Test with my own words
    word = "beautiful"
    one_hot = torch.Tensor(wordDic.getOneHot(word)).to(device)
    output = network(one_hot)
    print("{} is: {}".format(word, posDic.fromOneHot(output)))


if __name__ == "__main__":
    main()
