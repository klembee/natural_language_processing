import numpy as np
from torch.nn import MSELoss
from torch.optim import SGD
import torch
import os

from tests.part_of_speech_tagging.datasets import loadTestData, loadTrainData
from tests.part_of_speech_tagging.networks import POSTagger

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

    def getOneHot(self, word):
        one_hot = np.zeros(self.nb_words)
        if word in self.word2Index.keys():
            one_hot[self.word2Index[word]] = 1
        else:
            print("Word not in dictionary !")
        return one_hot

    def fromOneHot(self, one_hot):
        index = torch.argmax(one_hot)
        return self.words[index]


learning_rate = 10e-3
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

def main():
    wordDic = Dictionary()
    posDic = Dictionary()
    train_data = loadTrainData()
    test_data = loadTestData()

    nb_words = train_data['word'].nunique()
    nb_pos_tags = train_data['pos'].nunique()

    print("Creating dictionaries")
    for word in train_data['word']:
        wordDic.addWord(word)

    for pos in train_data['pos']:
        posDic.addWord(pos)

    network = POSTagger(nb_words, nb_pos_tags, device)

    if not os.path.isfile("model.pt"):
        # Converts the tra
        print("Converting training data to one hot")
        inputs = [wordDic.getOneHot(word) for word in train_data['word']]
        outputs = [posDic.getOneHot(pos) for pos in train_data['pos']]

        print("Training")
        loss_func = MSELoss()
        optimizer = SGD(network.parameters(), lr=learning_rate)

        network.train(inputs, outputs, loss_func, optimizer, "model.pt")
    else:
        print("Loaded from model saved file")
        network.load_state_dict(torch.load("model.pt"))

    # Testing the network
    inputs = [wordDic.getOneHot(word) for word in test_data['word']]
    outputs = [posDic.getOneHot(pos) for pos in test_data['pos']]
    network.test(inputs, outputs)

    # Test with my own words
    word = "beautiful"
    one_hot = torch.Tensor(wordDic.getOneHot(word)).to(device)
    output = network(one_hot)
    print("{} is: {}".format(word, posDic.fromOneHot(output)))



if __name__ == "__main__":
    main()
