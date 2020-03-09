import pandas as pd

train_file = "datasets/train.txt"
test_file = "datasets/test.txt"


def loadTrainData():
    """
    Loads the train dataset located in the datasets directory
    :return: a dataframe containing the training data
    """
    data = pd.read_csv(train_file, header=None, delim_whitespace=True)
    data.columns = ["word", "pos", "notimportant"]
    return data


def loadTestData():
    """
    Loads the testing dataset located in the datasets directory
    :return: a dataframe containing the testing data
    """
    data = pd.read_csv(test_file, header=None, delim_whitespace=True)
    data.columns = ["word", "pos", "notimportant"]
    return data
