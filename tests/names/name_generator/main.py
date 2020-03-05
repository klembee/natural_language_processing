# This code is inspired by https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
import sys
import torch
import string
import unicodedata
import torch.nn as nn
import random
import time
import math
import os

languages = ['arabic',
             'chinese',
             'czech',
             'dutch',
             'english',
             'french',
             'german',
             'greek',
             'irish',
             'italian',
             'japanese',
             'korean',
             'polish',
             'portuguese',
             'russian',
             'scottish',
             'spanish',
             'vietnamese']

# Will help to make one hot vectors representing letters
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

criterion = nn.NLLLoss()
learning_rate = 0.0005

def main(args):
    n_iters = 100000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0
    max_length = 20
    start = time.time()

    names = loadNames()
    network = RNN(len(languages), n_letters, 128, n_letters)

    # Load the model saved file if present
    if not os.path.isfile('savedModel.x'):
        for iter in range(1, n_iters + 1):
            output, loss = train(network, *randomtrainingExample(names))
            total_loss += loss

            if iter % print_every == 0:
                print('{}. {}/{}. Loss: {}'.format(timeSince(start), iter, n_iters, total_loss / plot_every))

            if iter % plot_every == 0:
                all_losses.append(total_loss / plot_every)
                total_loss = 0

        torch.save(network.state_dict(), "savedModel.x")
    else:
        network.load_state_dict(torch.load('savedModel.x'))

    with torch.no_grad():
        start_letter = "M"
        language = getLanguageTensor('russian')
        input = wordTensor(start_letter)
        hidden = network.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output_tensor, hidden = network(language, input[0], hidden)

            topv, topi = output_tensor.topk(1)
            if topi == n_letters - 1: #EOS token
                break;
            else:
                letter = all_letters[topi]
                output_name += letter
            input = wordTensor(letter)

        print(output_name)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class RNN(nn.Module):
    def __init__(self, nb_categories, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2o = nn.Linear(nb_categories + input_size + hidden_size, output_size)
        self.i2h = nn.Linear(nb_categories + input_size + hidden_size, hidden_size)
        self.o2o = nn.Linear(output_size + hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        output = self.i2o(input_combined)
        hidden = self.i2h(input_combined)
        out_combined = torch.cat((hidden, output), 1)
        output = self.o2o(out_combined)
        output = self.dropout(output)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def train(network, category_tensor, input_name_tensor, target_name_tensor):
    target_name_tensor.unsqueeze_(-1)
    hidden = network.initHidden()

    network.zero_grad()
    loss = 0

    for i in range(input_name_tensor.size(0)):
        # For each letters
        output, hidden = network(category_tensor, input_name_tensor[i], hidden)
        loss += criterion(output, target_name_tensor[i])

    loss.backward()

    for p in network.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_name_tensor.size(0)


def getLanguageTensor(language):
    languageIndex = languages.index(language)
    onehot = torch.zeros(1, len(languages))
    onehot[0][languageIndex] = 1

    return onehot


def wordTensor(word):
    onehot = torch.zeros(len(word), 1, n_letters)
    for i in range(len(word)):
        letter = word[i]
        onehot[i][0][all_letters.find(letter)] = 1

    return onehot


def getTargetTensor(word):
    letter_indices = [letter2index(word[i]) for i in range(1, len(word))]
    letter_indices.append(n_letters - 1)  # EOS token
    return torch.tensor(letter_indices, dtype=torch.long)


def randomTrainingPair(names):
    language = random.choice(languages)
    name = random.choice(names[language])

    return language, name


def randomtrainingExample(names):
    language, name = randomTrainingPair(names)
    language_tensor = getLanguageTensor(language)
    name_tensor = wordTensor(name)
    target_tensor = getTargetTensor(name)

    return language_tensor, name_tensor, target_tensor


def letter2index(letter):
    """
    :param letter: The letter to get the index of in the bag of letters
    :return: the index of the letter in the bag of letters
    """

    return all_letters.find(letter)


def nameToTensor(name):
    """
    :param name: the name to transform to a tensor
    :return: a tensor of size "name_length" x 1 x "nb_letters_in_bag_of_letters"
    """

    tensor = torch.zeros(len(name), 1, n_letters)
    for i, letter in enumerate(name):
        tensor[i][0][letter2index(letter)] = 1

    return tensor


def unicodeToAscii(s):
    """
    :param s: The string to convert to ascii
    :return: the string s with each character converted to ascii
    """

    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )


def loadNames():
    """
    Load the language names from the files in the data folder
    :return: a dictionary containing as keys the languages and as value
            an array of names
    """

    dir = "../data/names/"
    lang2names = {}
    for language in languages:
        # Open the file and load the data into the dictionary
        if not language in lang2names.keys():
            lang2names[language] = []

        with open(dir + language.capitalize() + ".txt", 'r', encoding="utf-8") as file:
            while True:
                name = file.readline()
                if not name:
                    break

                lang2names[language].append(unicodeToAscii(name))

    return lang2names


if __name__ == "__main__":
    main(sys.argv[1:])
