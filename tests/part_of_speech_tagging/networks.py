import torch.nn as nn
import torch


class POSTagger(nn.Module):
    """
    Simple neural network tagging a word without the context
    """

    def __init__(self, vocab_size, output_size, device):
        super(POSTagger, self).__init__()

        self.hidden2output = nn.Linear(vocab_size, output_size)
        self.softmax = nn.Softmax(dim=0)

        self.printEvery = 10000
        self.device = device
        self.to(device)

    def forward(self, one_hot_word):
        return self.hidden2output(one_hot_word)


class POSTaggerReccurent(nn.Module):

    def __init__(self, vocab_size, output_size, hidden_size, device):
        super(POSTaggerReccurent, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(vocab_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(vocab_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        next_hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        return output, next_hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)






