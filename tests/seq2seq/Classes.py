import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random
import unidecode

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20


# Normalize every sentences for a specific language
def normalize_sentence(df, lang):
    sentence = df[lang].str.lower()

    # Remove the accents
    sentence = sentence.map(lambda x: unidecode.unidecode(x).strip())

    # Remove the punctuations
    sentence = sentence.str.replace('[^A-Za-z\s]+', '')

    sentence = sentence.str.normalize('NFD')

    # todo: Find a way to normalize japanese since it cannot be converted to ascii
    # sentence = sentence.encode('ascii').str.decode('utf-8')

    return sentence


# Read and normalize the sentences of the two languages
# Returns the sentences in both language as a tuple: (sentences lang 1, sentences lang2)
def readSentences(df, lang1, lang2):
    sentences1 = normalize_sentence(df, lang1)
    sentences2 = normalize_sentence(df, lang2)
    return sentences1, sentences2


# Read the language data file and return a panda
# data frame containing the data
def read_file(loc, lang1, lang2):
    df = pd.read_csv(loc, delimiter='\t', lineterminator='\n', header=None, names=[lang1, lang2, 'attribution'])
    df = df[[lang1, lang2]]
    return df


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    indices = indexesFromSentence(lang, sentence)
    indices.append(EOS_token)
    return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], device)
    output_tensor = tensorFromSentence(output_lang, pair[1], device)

    return (input_tensor, output_tensor)

def calcModel(model, input_tensor, target_tensor, model_optimizer, criterion):
    model_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0

    output = model(input_tensor, target_tensor)
    num_iter = output.size(0)
    # print("Number of iterations: {}".format(num_iter))

    # Calculate the loss from a predicted sentence
    for ot in range(num_iter):
        loss += criterion(output[ot], target_tensor[ot])

    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter

    return epoch_loss

def trainModel(device, model, source, target, pairs, num_iteration=20000):
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    total_loss_iterations = 0

    # Choose the training pairs.
    # todo: Split the training and test data in a better way
    print("Getting training pairs")
    training_pairs = [tensorsFromPair(source, target, random.choice(pairs), device) for i in range(num_iteration)]

    for iter in range(1, num_iteration+1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_sensor = training_pair[1]

        loss = calcModel(model, input_tensor, target_sensor, optimizer, criterion)
        total_loss_iterations += loss

        if iter % 5000 == 0:
            average_loss = total_loss_iterations / 5000
            total_loss_iterations = 0
            print("Iteration: %d, Loss: %.4f" % (iter, average_loss))

    torch.save(model.state_dict(), 'trained.pt')
    return model

def evaluate(device, model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentences[0], device)
        output_tensor = tensorFromSentence(output_lang, sentences[1], device)

        decoded_words = []
        output = model(input_tensor, output_tensor)

        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(1)
            if topi[0].item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi[0].item()])

        return decoded_words

def evaluateRandomly(device, model, source, target, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print("Source: {}".format(pair[0]))
        print("Target: {}".format(pair[1]))
        output_words = evaluate(device, model, source, target, pair)
        print("Predicted: {}".format(' '.join(output_words)))

##
# Class that acts like a dictionary.
# You use it to add sentences which in turn adds words.
##
class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "ESO"}
        self.n_words = 2

    # Add a sentence to the dictionary
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    # Add a single word to the dictionary
    # and update the word count
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Init the embedding layer
        self.embedding = nn.Embedding(self.input_dim, self.embbed_dim)
        # Init the GRU layers
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)

    def forward(self, src):
        embedded = self.embedding(src).view(1, 1, -1)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()

        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Create the Embedding, GRU and linear layers
        self.embedding = nn.Embedding(output_dim, self.embbed_dim)
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output,hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))

        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # The number of words in the sentence
        input_length = source.size(0)
        batch_size = target.shape[1]
        target_length = target.shape[0]
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        # Encode every word in the sentence
        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(source[i])

        decoder_hidden = encoder_hidden.to(self.device)
        decoder_input = torch.tensor([SOS_token], device=self.device)

        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = (target[t] if teacher_force else topi)
            if(teacher_force == False and input.item() == EOS_token):
                break

        return outputs

