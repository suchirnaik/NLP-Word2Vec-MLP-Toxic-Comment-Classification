#To run this assignment please save all the csv and txt files in the same folder as this python file
#I have given relative paths for all files in this program

from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.neural_network import MLPClassifier

# Load pre-trained word embedding glove twitter 25
wv = KeyedVectors.load_word2vec_format('glove-twitter-25.txt', binary=False)

# The below function converts a comment to a matrix of word embeddings.
#It takes 2 arguments :
#Comment : the comment to convert
#max_length which is the maximum lenth of the comment
#The functon returns padded_embeddings which is a matrix of word embedding. These are padded to max_length with zero if necessary

def comment_to_embeddings(comment, max_length):
    words = comment.split()
    embeddings = []
    for word in words:
        try:
            embeddings.append(wv[word])
        except KeyError:
            pass
    if len(embeddings) == 0:
        return np.zeros((max_length, wv.vector_size))
    else:
        if len(embeddings) > max_length:
            embeddings = embeddings[:max_length]
        padded_embeddings = np.zeros((max_length, wv.vector_size))
        padded_embeddings[:len(embeddings), :] = embeddings
        return padded_embeddings

#The below function is to train the MLP model
#Its takes two arguments, one eing the path to the training data and the other being the number of hidden layer in the MLP model
# It returns the trained model   
def train_MLP_model(path_to_train_file, num_layers=2):
    # Load training data
    train_df = pd.read_csv(path_to_train_file)
    max_length = 40
    X_train = np.array([comment_to_embeddings(comment, max_length) for comment in train_df['comment_text']])
    X_train = X_train.reshape((X_train.shape[0], max_length, wv.vector_size))
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = train_df['toxic'].values

    # Train MLP model
    clf = MLPClassifier(hidden_layer_sizes=(100,) * num_layers, max_iter=100)
    clf.fit(X_train, y_train)

    return clf

#The below function is to test the MLP model on the test data
#It returns a dataframe with the predictons
def test_MLP_model(path_to_test_file, MLP_model):
    # Load test data
    test_df = pd.read_csv(path_to_test_file)
    max_length = 40
    X_test = np.array([comment_to_embeddings(comment,max_length) for comment in test_df['comment_text']])
    X_test = X_test.reshape((X_test.shape[0], max_length, wv.vector_size))
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Make predictions on test data
    y_prob = MLP_model.predict_proba(X_test)[:, 1]
    y_pred = MLP_model.predict(X_test)

    # Add predictions to test data and save to file
    test_df['toxic_prob'] = y_prob
    test_df['toxic_pred'] = y_pred
    test_df.to_csv('predictions.csv', index=False)

    return test_df

# The below lines of code train and test 1 layer MLP model and give the repsective scores
clf_1_layer = train_MLP_model('train.csv', num_layers=1)
test_df_1_layer = test_MLP_model('test.csv', clf_1_layer)
test_labels_df = pd.read_csv('test_labels.csv')
test_df_1_layer = pd.merge(test_df_1_layer, test_labels_df, on='id', how='inner')
test_df_1_layer = test_df_1_layer[test_df_1_layer['toxic'] != -1]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_test_1_layer = test_df_1_layer['toxic'].values
y_pred_1_layer = test_df_1_layer['toxic_pred'].values

accuracy = accuracy_score(y_test_1_layer, y_pred_1_layer)
precision = precision_score(y_test_1_layer, y_pred_1_layer)
recall = recall_score(y_test_1_layer, y_pred_1_layer)
f1 = f1_score(y_test_1_layer, y_pred_1_layer)


print('Accuracy for 1 layer MLP model:', accuracy)
print('Precision for 1 layer MLP model:', precision)
print('Recall for 1 layer MLP model:', recall)
print('F1 score for 1 layer MLP model:', f1)

from sklearn.metrics import accuracy_score

y_test = test_df_1_layer['toxic'].values
y_pred = test_df_1_layer['toxic_pred'].values

# Split the arrays based on class label
toxic_mask = y_test == 1
non_toxic_mask = y_test == 0

# Calculate accuracy score for toxic and non-toxic classes separately
toxic_acc = accuracy_score(y_test[toxic_mask], y_pred[toxic_mask])
non_toxic_acc = accuracy_score(y_test[non_toxic_mask], y_pred[non_toxic_mask])

print('Accuracy of toxic class for 1 layer MLP model:', toxic_acc)
print('Accuracy of non-toxic class for 1 layer MLP model:', non_toxic_acc)

#  The below lines of code train and test 2 layer MLP model and give the respective scores
clf_2_layer = train_MLP_model('train.csv')
test_df_2_layer = test_MLP_model('test.csv', clf_2_layer)
test_df_2_layer = pd.merge(test_df_2_layer, test_labels_df, on='id', how='inner')
test_df_2_layer = test_df_2_layer[test_df_2_layer['toxic'] != -1]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_test_2_layer = test_df_2_layer['toxic'].values
y_pred_2_layer = test_df_2_layer['toxic_pred'].values

accuracy = accuracy_score(y_test_2_layer, y_pred_2_layer)
precision = precision_score(y_test_2_layer, y_pred_2_layer)
recall = recall_score(y_test_2_layer, y_pred_2_layer)
f1 = f1_score(y_test_2_layer, y_pred_2_layer)


print('Accuracy of 2 layer MLP model:', accuracy)
print('Precision 2 layer MLP model:', precision)
print('Recall 2 layer MLP model:', recall)
print('F1 score 2 layer MLP model:', f1)

from sklearn.metrics import accuracy_score

y_test = test_df_2_layer['toxic'].values
y_pred = test_df_2_layer['toxic_pred'].values

# Split the arrays based on class label
toxic_mask = y_test == 1
non_toxic_mask = y_test == 0

# Calculate accuracy score for toxic and non-toxic classes separately
toxic_acc = accuracy_score(y_test[toxic_mask], y_pred[toxic_mask])
non_toxic_acc = accuracy_score(y_test[non_toxic_mask], y_pred[non_toxic_mask])

print('Accuracy of toxic class for 2 layer MLP model:', toxic_acc)
print('Accuracy of non-toxic class 2 layer MLP model:', non_toxic_acc)

# The below lines of code train and test 3 layer MLP model and give the respective scores
clf_3_layer = train_MLP_model('train.csv', num_layers=3)
test_df_3_layer = test_MLP_model('test.csv', clf_3_layer)
test_df_3_layer = pd.merge(test_df_3_layer, test_labels_df, on='id', how='inner')
test_df_3_layer = test_df_3_layer[test_df_3_layer['toxic'] != -1]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_test_3_layer = test_df_3_layer['toxic'].values
y_pred_3_layer = test_df_3_layer['toxic_pred'].values

accuracy = accuracy_score(y_test_3_layer, y_pred_3_layer)
precision = precision_score(y_test_3_layer, y_pred_3_layer)
recall = recall_score(y_test_3_layer, y_pred_3_layer)
f1 = f1_score(y_test_3_layer, y_pred_3_layer)


print('Accuracy for 3 layer MLP model:', accuracy)
print('Precision for 3 layer MLP model:', precision)
print('Recall for 3 layer MLP model:', recall)
print('F1 score for 3 layer MLP model:', f1)

from sklearn.metrics import accuracy_score

y_test = test_df_3_layer['toxic'].values
y_pred = test_df_3_layer['toxic_pred'].values

# Split the arrays based on class label
toxic_mask = y_test == 1
non_toxic_mask = y_test == 0

# Calculate accuracy score for toxic and non-toxic classes separately
toxic_acc = accuracy_score(y_test[toxic_mask], y_pred[toxic_mask])
non_toxic_acc = accuracy_score(y_test[non_toxic_mask], y_pred[non_toxic_mask])

print('Accuracy of toxic class for 3 layer MLP model:', toxic_acc)
print('Accuracy of non-toxic class for 3 layer MLP model:', non_toxic_acc)
