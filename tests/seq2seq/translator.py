from __future__ import unicode_literals, print_function, division
import torch

from tests.seq2seq.Classes import Lang, \
    read_file, readSentences, MAX_LENGTH, Encoder, Decoder, \
    evaluateRandomly, trainModel, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read the translation file
df = read_file("fra.txt", "eng", "fra")
print(df.head())
print("Number of entries: {}".format(len(df)))

sentences1, sentences2 = readSentences(df, "eng", "fra")

source = Lang()
target = Lang()

# Create the pairs
pairs = []
for i in range(0,100): # len(df)
    if len(sentences1[i].split(' ')) < MAX_LENGTH and len(sentences2[i].split(' ')) < MAX_LENGTH:
        source.addSentence(sentences1[i])
        target.addSentence(sentences2[i])
        pairs.append([sentences1[i], sentences2[i]])

# Train the model

input_size = source.n_words
output_size = target.n_words

embed_size = 256
hidden_size = 512
num_layers = 1
num_iteration = 100000

encoder = Encoder(input_size, hidden_size, embed_size, num_layers)
decoder = Decoder(output_size, hidden_size, embed_size, num_layers)

model = Seq2Seq(encoder, decoder, device).to(device)

print("\nTraining")
model = trainModel(device, model, source, target, pairs, num_iteration)

print("\nEvaluating")
evaluateRandomly(device, model, source, target, pairs)