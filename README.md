# Natural language processing

In this repository, I am sharing all my tests with natural language processing. I am actually learning this field and would appreciate feedback !

## How to run the examples
1. Create a new conda environment: `conda env create -n nlp_tests`
1. Go into this new environment: `activate nlp_tests`
1. Install the requirements: `pip install -r requirements.txt`
1. Download the datasets used in my examples by running: `python install_datasets.py`

## File structure
I put my small tests in the `tests/` directory. When I am making a bigger project, I put it in `projects/`

## Tests

### BERT
In this directory, I am playing with BERT network architecture using the package transformers. I used this network artitechture for sentiment analysis and text question extraction.

### GloVe
In this directory, I am playing with the GloVe word embeding algorithm. I try to build my own word embeding matrix from scratch

### Names
In this directory, I am working a name generator and a name classifier. I built a neural network that learns to classify where a given name is originating from. I also did the reverse task to generating a name given an origin.

### Part of speech tagging
In this directory, I am trying to tag words in a sentence with their corresponding POS tag.

### Seq2Seq
In this directory, I am playing with seq2seq network architecture. Using this architecture, I tried to make a translator from French to English. (It doesn't work well for now, I will have to come back later)

### Subreddit classifier
In this directory, I am trying to build a network able to guess from which subreddit a Reddit post was posted on.

### T5
In this directory, I am playing with the T5 architecture, proposed by Google in 2019.

### Word2vec
In this directory, I am building a word2vec matrix from scratch.
