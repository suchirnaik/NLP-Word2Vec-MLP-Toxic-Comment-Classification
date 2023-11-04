# NLP Project: Toxic Comment Classification with Word2Vec and MLPs

## Overview
Welcome to my NLP project focusing on text classification using Word2Vec embeddings and Multi-Layer Perceptrons (MLPs). This project explores the application of pre-trained Word2Vec embeddings in combination with MLPs for classifying toxic comments. I used the toxic comment dataset, which you might remember from previous assignments.

### Useful Python Packages
To perform this classification task, I utilized various Python packages. Here are the key ones:

- [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) - Gensim is a handy library for working with Word2Vec embeddings. It also provides access to pre-trained Word2Vec models.
- [scikit-learn (sklearn)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) - For implementing MLPs, I used the `MLPClassifier` from sklearn.

### Dataset
The dataset used for this project is the same toxic comment dataset that you encountered in previous assignments (HW3-4). This dataset contains comments, some of which are toxic. Please exercise caution when working with the data.

You can access the training and test sets [here](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data). For the Word2Vec tasks, I primarily focused on the toxic label, where toxic=1 and non-toxic=0.

## Final Report
Similar to previous assignments, I documented my observations and results in a final report.

## MLP Tasks
The key tasks for this project involved implementing Multi-Layer Perceptrons (MLPs) and applying them to text classification using Word2Vec embeddings. Here are the primary tasks I performed:

### Task 1: `train_MLP_model(path_to_train_file, num_layers = 2)`
This function trains an MLP model on the training data and returns the trained model. The training texts are represented using Word2Vec embeddings. I had the flexibility to choose any pre-trained Word2Vec embeddings. The input size affects how much of the text can be sent into the model, and I made informed decisions about this. The training data followed the same format as the training data file.

### Task 2: `test_MLP_model(path_to_test_file, MLP_model)`
This function tests a trained MLP model on a test file and produces predictions for all test texts. It generates a test file with two added columns: 1) the probability of the text being toxic, and 2) the class prediction (toxic or not toxic). The format for the input file matched the test file format.

Once these functions were implemented, I proceeded with the following tasks:

1. Trained a 2-layer MLP model on the entire train set.
2. Tested the trained model on the test set and generated predictions for all test texts.
3. Repeated steps 1 and 2 for 1-layer and 3-layer MLPs.
4. Analyzed and compared the overall accuracies of the models. I made sure to report any changes that led to accuracy improvements or drops.

My analysis and comparison observations are presented in the report document.

