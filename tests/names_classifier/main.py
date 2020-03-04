# Code inspired from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
import sys
import string
import unicodedata
import torch
import torch.nn as nn
import random
import time
import math
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

learning_rate = 0.005
n_iters = 100000
print_every = 5000
plot_every = 1000

device = torch.device('cpu')
# if torch.cuda.is_available():
# device = torch.device('cuda:0')

criterion = nn.NLLLoss().to(device)


def main(args):
    # Create the network
    network = RNN(n_letters, len(languages)).to(device)
    names = loadNames()

    # Train the model
    current_loss = 0
    all_losses = []
    start = time.time()

    for iter in range(1, n_iters + 1):
        language, name, language_tensor, name_tensor = randomTrainingExample(names)
        output, loss = train(network, criterion, language_tensor, name_tensor)

        current_loss += loss

        if iter % print_every == 0:
            guess, guess_i = network.categoryFromOutput(output)
            correct = 'YES' if guess == language else "NO {}".format(language)
            print('{}/{}. Loss: {} Time since start: {}'.format(iter, n_iters, current_loss / plot_every,
                                                                timeSince(start)))

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    # Evaluate the model
    confusion = torch.zeros(len(languages), len(languages))
    n_confusion = 10000

    for i in range(n_confusion):
        language, name, language_tensor, name_tensor = randomTrainingExample(names)
        output = evaluate(network, name_tensor)
        guess, guess_i = network.categoryFromOutput(output)
        language_i = languages.index(language)
        confusion[language_i][guess_i] += 1

    # Normalize every confusion lines
    for i in range(len(languages)):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Display confusion matrix in plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + languages, rotation=90)
    ax.set_yticklabels([''] + languages)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

    # Ask the user for names
    while True:
        try:
            print("What name should I classify ?")
            name = input()
            output = evaluate(network, nameToTensor(name))
            print("My guess: {}".format(network.categoryFromOutput(output)))
        except:
            print("Name doesnt exist in vocabulary")


class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = 256

        self.i2h = nn.Linear(input_size + self.hidden_size, self.hidden_size)
        self.i2o = nn.Linear(input_size + self.hidden_size, 50)

        self.i2o2 = nn.Linear(50, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = F.logsigmoid(self.i2o2(output))
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def categoryFromOutput(self, output):
        """
        :param output: the output of the neural network
        :return: the language that the output suggests match the most
        """

        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()

        return languages[category_i], category_i


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(network, criterion, category_tensor, name_tensor):
    hidden = network.initHidden().to(device)

    network.zero_grad()

    for i in range(name_tensor.size()[0]):
        # Loop through each letter of the name
        name_tensor = name_tensor.to(device)
        output, hidden = network(name_tensor[i], hidden)

    loss = criterion(output, category_tensor.to(device))
    loss.backward()

    for p in network.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


def evaluate(network, name_tensor):
    hidden = network.initHidden()

    for i in range(name_tensor.size()[0]):
        output, hidden = network(name_tensor[i], hidden)

    return output


def randomTrainingExample(names):
    """
    :param names: Get a random training example
    :return: a random name from a random language
    """
    category = random.choice(languages)
    name = random.choice(names[category])

    category_tensor = torch.tensor([languages.index(category)], dtype=torch.long)
    name_tensor = nameToTensor(name)
    return category, name, category_tensor, name_tensor


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

    dir = "./data/names/"
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
