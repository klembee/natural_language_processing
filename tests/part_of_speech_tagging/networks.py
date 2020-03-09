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

    def train(self, x, y, loss_func, optimizer, model_save_file):

        total_loss = 0

        for i in range(len(x)):
            word = torch.Tensor(x[i]).to(self.device)
            target = torch.Tensor(y[i]).to(self.device)

            optimizer.zero_grad()

            prediction = self(word)
            loss = loss_func(prediction, target)
            total_loss += loss

            # print()
            # print(prediction)
            # print(target)

            loss.backward()
            optimizer.step()

            if i % self.printEvery == 0:
                print("{}/{}: {}".format(i, len(x), total_loss/self.printEvery))
                total_loss = 0

        torch.save(self.state_dict(), model_save_file)

    def test(self, x, y):
        """
        Test the network with the provided inputs and outputs
        :param x: the inputs to give to the network
        :param y: the expected outputs of the network
        """

        nb_rights = 0


        for i in range(len(x)):
            input = torch.Tensor(x[i]).to(self.device)
            output = torch.Tensor(y[i]).to(self.device)

            prediction = self(input)

            pred_highest_index = torch.argmax(prediction)
            output_highest_index = torch.argmax(output)

            if(pred_highest_index == output_highest_index):
                nb_rights += 1

            if (i + 1) % self.printEvery == 0:
                print("{} %".format((nb_rights/i) * 100))






