# ======================================================================================================================
# Import all required libraries and modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import itertools
from time import time
import datetime
from random import randint
from statistics import mean, stdev
import gensim
import zipfile
import spacy
from bs4 import BeautifulSoup
import unicodedata
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pickle

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn import svm

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
from keras.wrappers.scikit_learn import KerasClassifier

# ======================================================================================================================
# Loading Quora Dataset and Normalising Questions

path = ""
x_labels = ["question1", "question2"]  # Column labels used for data in deep learning models
svm_features = ["words", "labels"]  # column labels used for constituent features
y_label = "is_duplicate"
embedding_dim = 300
nlp = spacy.load('en_core_web_sm')
nltk.download('punkt')
nltk.download('stopwords')
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}


def load_data(data_dir):
    """
    LOAD DATA

    Parameters
    ----------
    data_dir : directory path for data to be loaded

    """
    df = pd.read_csv(data_dir)
    return df


def remove_class_imbalance(df, ratio):
    """
    remove class imbalance

    Parameters
    ----------
    df : dataframe with class imbalance
    ratio : required ratio between classes

    """
    X = df[x_labels]
    y = df[y_label]
    counter1 = Counter(y)  # count number of each class label
    print("Current class imbalance:")
    print(counter1)
    under_sampler = RandomUnderSampler(sampling_strategy=ratio)  # create under-sampler
    X_under, y_under = under_sampler.fit_resample(X, y)
    counter2 = Counter(y_under)
    print("New class imbalance:")
    print(counter2)
    new_df = pd.concat([X_under.reset_index(drop=True), y_under.reset_index(drop=True)], axis=1)  # create new dataframe
    return new_df


def remove_nan_rows(df):
    """
    remove nan rows

    Parameters
    ----------
    df : dataframe with nan rows

    """
    is_nan = df.isnull()
    row_has_nan = is_nan.any(axis=1)
    rows_with_nan = df[row_has_nan]
    print("A total of {} NaN rows removed.".format(len(rows_with_nan)))
    return df.dropna()


def remove_accented_chars(text):
    """
    remove accented characters

    Parameters
    ----------
    text : text with accented characters

    """
    # use NFKD normiliser to replace accented characters from text
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text, contraction_mapping):
    """
    expand contractions

    Parameters
    ----------
    text : text with contractions
    contraction_mapping: dictionary of contraction and expansion pairs

    """

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)  # replace all contractions with their expansions
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_special_characters(text, remove_digits=False):
    """
    remove special characters and digits

    Parameters
    ----------
    text : text with special characters and digits
    remove_digits : By default false, used to specify whether or not to remove digits as well

    """

    pattern = r"[^A-Za-z0-9^,!.\/'+-=]" if not remove_digits else r"[^A-Za-z^,!.\/'+-=]"  # define characters to be replaced
    text = re.sub(pattern, " ", text)
    return text


def lemmatize_text(text):
    """
    lemmatize text

    Parameters
    ----------
    text : text to be lemmatized

    """
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])  # replace words with their lemma
    return text


def remove_stopwords(text, is_lower_case=False):
    """
    remove stopwords

    Parameters
    ----------
    text : text with stopwords
    is_lower_case : By default false, used to specify text already lowercase or not

    """
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]  # separate text into its tokens
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]  # filter tokens on whether they exist in the stopword list
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list] # filter tokens on whether they exist in the stopword list
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True,
                     text_lemmatization=True, special_char_removal=True,
                     stopword_removal=True, remove_digits=True):
    """
    normalise all questions

    Parameters
    ----------
    corpus : questions to be normalised
    contraction_expansion : By default true, used to specify whether ot not to perform contraction expansion
    accented_char_removal : By default true, used to specify whether ot not to perform accented char removal
    text_lower_case : By default true, used to specify whether ot not to convert text to lower case
    text_lemmatization : By default true, used to specify whether ot not to perform text lemmatization
    special_char_removal : By default true, used to specify whether ot not to perform special char removal
    stopword_removal : By default true, used to specify whether ot not to perform stopword removal
    remove_digits : By default true, used to specify whether ot not to remove digits with special characters
    """

    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc, CONTRACTION_MAP)
        # lowercase the text
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r"[\r|\n|\r\n]+", " ", doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove special characters and\or digits
        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)
            # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)

        normalized_corpus.append(doc)

    return normalized_corpus


def normalise_dataframe(df):
    """
    normalise dataframe of question pairs

    Parameters
    ----------
    df : dataframe of question pairs

    """
    print('Normalising corpus')
    columns = x_labels + [y_label]
    new_df = pd.DataFrame(columns=columns)
    new_df['question1'] = normalize_corpus(corpus=df[x_labels[0]])
    new_df['question2'] = normalize_corpus(corpus=df[x_labels[1]])
    new_df['is_duplicate'] = df[y_label].values
    return new_df.reset_index(drop=True)


def sample_data(df, frac):
    """
    sample data

    Parameters
    ----------
    df : dataframe of question pairs
    frac : fraction of data to be sampled

    """
    return df.sample(frac=frac)


# ======================================================================================================================
#  Create Constituent Features


def jaccard_similarity(list1, list2):
    """
    jaccard similarity helper function

    Parameters
    ----------
    list1 : list of values to be compared
    list2 : list of values to be compared

    """
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2)) + 1


def create_word_features(dict1, dict2):
    """
    create constituent word features

    Parameters
    ----------
    dict1 : dictionary of label and word pairs from question 1
    dict2 : dictionary of label and word pairs from question 2

    """
    keyset1 = list(dict1.keys())
    keyset2 = list(dict2.keys())
    all_labels = list(dict.fromkeys(keyset1 + keyset2))
    all_labels_dict = []
    for i in all_labels:
        if ((i in keyset1) and (i in keyset2)):
            jac_sim = jaccard_similarity(dict1[i], dict2[i])
        else:
            jac_sim = -1
        all_labels_dict.append(jac_sim)

    return mean(all_labels_dict)


def create_label_features(dict1, dict2):
    """
    create constituent label features

    Parameters
    ----------
    dict1 : dictionary of label and word pairs from question 1
    dict2 : dictionary of label and word pairs from question 2

    """
    keyset1 = list(dict1.keys())
    keyset2 = list(dict2.keys())
    if (len(keyset1) == 0 and len(keyset2) == 0):
        return 0
    elif (len(keyset1) == 0 or len(keyset2) == 0):
        return -1
    else:
        return jaccard_similarity(keyset1, keyset2)


def get_constituent_features(df):
    """
    create constituent features

    Parameters
    ----------
    df : dataframe of question pairs

    """
    feat_list = []
    start_time = time()
    print('Creating constituent features:')
    for i in range(len(df)):
        # create constituent trees from questions as string
        prediction1 = predictor.predict_json({"sentence": df[x_labels[0]].values[i]})
        prediction2 = predictor.predict_json({"sentence": df[x_labels[1]].values[i]})
        is_duplicate = df[y_label].values[i]

        # create NLTK trees for easier manipulation of constituent values
        tree1 = nltk.Tree.fromstring(prediction1['trees'])
        tree2 = nltk.Tree.fromstring(prediction2['trees'])

        q1_labels = {}
        q2_labels = {}
        # only show the most basic constituents (only the leaves of the tree)
        for j in tree1.subtrees(filter=lambda t: t.height() == 2):
            if j.label() in q1_labels.keys():
                q1_labels[j.label()] = q1_labels[j.label()] + [j[0]]
            else:
                q1_labels[j.label()] = [j[0]]

        for j in tree2.subtrees(filter=lambda t: t.height() == 2):
            if j.label() in q2_labels.keys():
                q2_labels[j.label()] = q2_labels[j.label()] + [j[0]]
            else:
                q2_labels[j.label()] = [j[0]]

        features = {}
        features[svm_features[0]] = create_word_features(q1_labels, q2_labels)
        label_fet = create_label_features(q1_labels, q2_labels)
        features[svm_features[1]] = label_fet
        features[y_label] = is_duplicate

        feat_list.append(features)
    print("Time taken:{}".format(datetime.timedelta(seconds=time() - start_time)))
    return feat_list


# ======================================================================================================================
# Creating Word Embeddings, Train-Validation-Test Split

archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
predictor = Predictor.from_archive(archive, 'constituency-parser')


def load_w2v_model():
    """ load word2vec model """
    return gensim.models.KeyedVectors.load_word2vec_format(path + 'GoogleNews-vectors-negative300.bin.gz',
                                                           binary=True)


def delete_w2v_model(model):
    """
    delete word2vec model
    Parameters
    ----------
    model : word2vec model to be deleted

    """
    del model


def create_vocabulary(df, word2vec):
    """
    create vocabulary and inverse vocabulary

    Parameters
    ----------
    df : dataframe of question pairs
    word2vec : word2vec model loaded

    """
    # Prepare embedding
    print('Creating vocabulary...')
    vocab = dict()
    # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
    inverse_vocab = ['<unk>']
    # Iterate over the questions
    for dataset in [df]:
        for index, row in dataset.iterrows():
            # Iterate through the text of both questions of the row
            for question in x_labels:
                q2n = []  # q2n -> question numbers representation
                for word in row[question].split():
                    if word not in word2vec.vocab:  # if word is not in the vocabulary of the word2vec model it is disregarded as it will not have an embeddign for it
                        continue
                    if word not in vocab:
                        vocab[word] = len(inverse_vocab)
                        q2n.append(len(inverse_vocab))
                        inverse_vocab.append(word)
                    else:
                        q2n.append(vocab[word])
                # Replace questions as word to question as number representation
                dataset.at[index, question] = q2n
    print('Vocabulary Created')
    return vocab, inverse_vocab, df


def create_embedding_matrix(vocab, word2vec, dim=300):
    """
    create embeddings matrix

    Parameters
    ----------
    vocab : vocabulary created from all questions in dataframe
    word2vec : the word2vec model loaded
    dim : By default 300, specified the embedding dimension to be used
    """
    print('Creating embedding matrix...')
    emb = 1 * np.random.randn(len(vocab) + 1, dim)  # This will be the embedding matrix
    emb[0] = 0  # So that the padding will be ignored
    # Build the embedding matrix
    for word, index in vocab.items():
        if word in word2vec.vocab:
            emb[index] = word2vec.word_vec(word)  # creating word embeddings
    print('Embedding Matrix Created')
    return emb


def train_val_test_split(df, max_seq_length):
    """
    split data for deep learning models

    Parameters
    ----------
    df : dataframe of question pairs
    max_seq_length : the longest question length in dataframe

    """
    print('Splitting data:')

    X = df[x_labels]
    Y = df[y_label]

    X_tr_full, X_tst, Y_tr_full, Y_tst = train_test_split(X, Y, test_size=0.1)  # split test and train
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_tr_full, Y_tr_full, test_size=0.1)  # split train and validation

    # Split to dicts
    X_tr = {x_labels[0]: X_tr.question1, x_labels[1]: X_tr.question2}
    X_val = {x_labels[0]: X_val.question1, x_labels[1]: X_val.question2}
    X_tst = {x_labels[0]: X_tst.question1, x_labels[1]: X_tst.question2}

    # Convert labels to their numpy representations
    Y_tr = Y_tr.values
    Y_val = Y_val.values
    Y_tst = Y_tst.values

    # Zero padding
    for dataset, side in itertools.product([X_tr, X_val, X_tst], x_labels):
        # pad values such that they a
        # re all the same length using the max sequence length
        dataset[side] = keras.preprocessing.sequence.pad_sequences(dataset[side], maxlen=max_seq_length)

    # Make sure everything is ok
    assert X_tr[x_labels[0]].shape == X_tr[x_labels[1]].shape
    assert len(X_tr[x_labels[0]]) == len(Y_tr)

    return X_tr, X_val, X_tst, Y_tr, Y_val, Y_tst


# ======================================================================================================================
# Create Ensemble train-validation-test split


def ensemble_train_val_test_split(svm_df, nn_df):
    """
    split data for ensmeble model

    Parameters
    ----------
    svm_df : dataframe of word and labels features
    nn_df : dataframe of question pairs

    """
    msk1_length = min(len(svm_df), len(nn_df))
    msk1 = np.random.rand(msk1_length) < 0.9  # create mask for train, and test split

    full_train_svm = svm_df[msk1]
    test_svm = svm_df[~msk1]
    full_train_nn = nn_df[msk1]
    test_nn = nn_df[~msk1]

    msk2_length = min(len(full_train_svm), len(full_train_nn))
    msk2 = np.random.rand(msk2_length) < 0.9  # create mask for train and validation split

    train_svm = full_train_svm[msk2]
    val_svm = full_train_svm[~msk2]
    train_nn = full_train_nn[msk2]
    val_nn = full_train_nn[~msk2]

    return full_train_nn, full_train_svm, train_nn, train_svm, val_nn, val_svm, test_nn, test_svm


def reformat_train_val_test(train, val, test, format_type, sampled_max_seq_length):
    """
    reformat data used for ensemble model based on sub-model

    Parameters
    ----------
    train : training data to be reformated to the correct format for use
    val : validation data to be reformated to the correct format for use
    test : testing data to be reformated to the correct format for use
    format_type : format all data should be converted to
    sampled_max_seq_length : longest question length
    """
    columns = list(train.columns)
    if (format_type == 'NN'):
        # Formatting for deep learning models
        x_tr = {x_labels[0]: train[columns[0]], x_labels[1]: train[columns[1]]}
        x_val = {x_labels[0]: val[columns[0]], x_labels[1]: val[columns[1]]}
        x_tst = {x_labels[0]: test[columns[0]], x_labels[1]: test[columns[1]]}

        # Convert labels to their numpy representations
        y_tr = train[columns[2]]
        y_val = val[columns[2]]
        y_tst = test[columns[2]]

        # Zero padding
        for dataset, side in itertools.product([x_tr, x_val, x_tst], x_labels):
            dataset[side] = keras.preprocessing.sequence.pad_sequences(dataset[side], maxlen=sampled_max_seq_length)

        # Make sure everything is ok
        assert x_tr[x_labels[0]].shape == x_tr[x_labels[1]].shape
        assert len(x_tr[x_labels[0]]) == len(y_tr)
    else:
        # formatting for SVM model
        x_tr = pd.concat([train[columns[0]], train[columns[1]]], axis=1, keys=svm_features)
        x_val = pd.concat([val[columns[0]], val[columns[1]]], axis=1, keys=svm_features)
        x_tst = pd.concat([test[columns[0]], test[columns[1]]], axis=1, keys=svm_features)

        # Convert labels to their numpy representations
        y_tr = train[columns[2]]
        y_val = val[columns[2]]
        y_tst = test[columns[2]]

        # Make sure everything is ok
        assert x_tr[svm_features[0]].values.shape == x_tr[svm_features[1]].shape
        assert len(x_tr[svm_features[0]]) == len(y_tr)

    return x_tr, x_val, x_tst, y_tr, y_val, y_tst


# ======================================================================================================================
# Build Si-Bi-LSTM Model


def exponent_neg_manhattan_distance(left, right):
    """
    Helper function to calculate the manhattan distance of the LSTMs outputs

    Parameters
    ----------
    left : output of left tower of LSTM
    right : output of right tower of LSTM

    """
    dist = K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))
    return dist


def create_concat_feature_vector(left, right):
    """
    Helper function to create concatenation vector of the LSTMs outputs

    Parameters
    ----------
    left : output of left tower of LSTM
    right : output of right tower of LSTM

    """
    v1 = left
    v2 = right
    diff = left - right
    v3 = K.square(diff)
    v4 = tf.keras.layers.Multiply()([left, right])  # Hadamard product / element wise
    v = K.concatenate([v1, v2, v3, v4])
    return v


def create_model(lstm_layer_size, dense_layer_size1, dense_layer_size2, grad_clip_norm,
                 lr, l1, l2, output_type, max_seq_length, embeddings):
    """
    Build layers of DL network

    Parameters
    ----------
    lstm_layer_size : sets size of lstm layer
    dense_layer_size1 : sets size of dense layer 1
    dense_layer_size2 : sets size of dense layer 2
    grad_clip_norm : sets gradient clipping of optimiser
    lr : sets learning rate of optimiser
    l1 : sets l1 regulariser value
    l2 : sets l2 regulariser value
    output_type : defines similarity measure used
    max_seq_length : maximum length of question
    embeddings : embeddings matrix created for embedding layer
    """
    print('Compiling model:')
    # The input layers
    left_input = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32')
    right_input = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32')
    embedding_layer = tf.keras.layers.Embedding(len(embeddings), embedding_dim, weights=[embeddings],
                                                input_length=max_seq_length, trainable=False)
    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    print('Creating Bi-LSTM towers')
    shared_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_layer_size))
    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    if (output_type == 'dist'):
        print('Calculating Manhattan distance')
        # loss function used for manhattan distance variant
        loss = 'mean_squared_error'
        # Calculates the distance as defined by the MaLSTM model
        output_layer = tf.keras.layers.Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                                              output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
    elif (output_type == 'NN'):
        # Create concat vecor to use as inout to fully connect NN
        print('Creating similairty network')
        # loss function used for feed-forward network variant
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        reg = keras.regularizers.l1_l2(l1=l1, l2=l2)
        concat_feature_vector = tf.keras.layers.Lambda(function=lambda x:
        create_concat_feature_vector(x[0], x[1]))([left_output, right_output])
        dense_layer1 = tf.keras.layers.Dense(dense_layer_size1, activation='selu', kernel_initializer='lecun_normal',
                                             kernel_regularizer=reg)(concat_feature_vector)
        dense_layer2 = tf.keras.layers.Dense(dense_layer_size2, activation='selu', kernel_initializer='lecun_normal',
                                             kernel_regularizer=reg)(dense_layer1)
        output_layer = tf.keras.layers.Dense(2, activation='softmax')(dense_layer2)
    else:
        print("Output type not recognised.")
        assert False

    # Pack it all up into a model
    model = tf.keras.models.Model([left_input, right_input], [output_layer])
    # Nadam optimizer, with gradient clipping by norm
    optimizer = tf.keras.optimizers.Nadam(lr=lr, clipnorm=grad_clip_norm)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def train_model(model, x_tr, y_tr, x_val, y_val, model_name, save_model, batch_size, n_epoch):
    """
    train DL model

    Parameters
    ----------
    model : model to be trained
    x_tr : training data inputs
    y_tr : training data labels
    x_val : validation data inputs
    y_val : validation data labels
    model_name : defines model name
    save_model : sets whether the model should be saved or not
    batch_size : defines bacth size used
    n_epoch : defines max number of training epochs

    """
    training_start_time = time()
    print('Training started:')
    # define early stopping criteria
    stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', min_delta=0.01)
    # define LR scheduler
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', min_lr=0.00001)

    if (save_model):
        checkpoint = ModelCheckpoint(path + model_name + ".hdf5", monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='auto')
        trained_model = model.fit([x_tr[x_labels[0]], x_tr[x_labels[1]]], y_tr, batch_size=batch_size, epochs=n_epoch,
                                  validation_data=([x_val[x_labels[0]], x_val[x_labels[1]]], y_val),
                                  callbacks=[checkpoint, stop, reduce_lr])
    else:
        trained_model = model.fit([x_tr[x_labels[0]], x_tr[x_labels[1]]], y_tr, batch_size=batch_size, epochs=n_epoch,
                                  validation_data=([x_val[x_labels[0]], x_val[x_labels[1]]], y_val),
                                  callbacks=[stop, reduce_lr])

    print("Training time finished.\n{} epochs in {}".format(n_epoch,
                                                            datetime.timedelta(seconds=time() - training_start_time)))

    return trained_model


# ======================================================================================================================
# Hyper-parameter Tuning for Si-Bi-LSTM models

# optimiser parameters
lr_array = [1e-3, 1e-4]
decay_array = [1e-6, 1e-9, 0]
grad_clip_norm_array = [0.75, 1, 1.25]

# neurons in each layer
nn1_array = [32, 64, 128]
nn2_array = [32, 64, 128]
nn3_array = [32, 64, 128]

# regularisation
l1_array = [0, 0.01, 0.001, 0.0001]
l2_array = [0, 0.01, 0.001, 0.0001]

# batch size
batch_array = [32, 64, 128]


def hyper_parameter_tuning(x_train, y_train, x_validation, y_validation, num_iterations, output_type, embeddings,
                           max_seq_length):
    """
    tune hyper-parameters of DL model

    Parameters
    ----------
    x_train : training data inputs
    y_train : training data labels
    x_validation : validation data inputs
    y_validation : validation data labels
    num_iterations : number of tuning iterations
    output_type : similairty measure used
    embeddings : embedding matrix used
    max_seq_length : maximum length of questions
    """
    best_acc = -1
    selected_parameters = []
    for i in range(num_iterations):
        # randomly set index values
        r1 = randint(0, 1)
        r2 = randint(0, 2)
        r3 = randint(0, 2)
        r4 = randint(0, 2)
        r5 = randint(0, 2)
        r6 = randint(0, 2)
        r7 = randint(0, 3)
        r8 = randint(0, 3)
        r9 = randint(0, 2)

        # set selected parameters
        lr = lr_array[r1]
        decay = decay_array[r2]
        grad_clip_norm = grad_clip_norm_array[r3]
        nn1 = nn1_array[r4]
        nn2 = nn2_array[r5]
        nn3 = nn3_array[r6]
        l1 = l1_array[r7]
        l2 = l2_array[r8]
        batch_size = batch_array[r9]

        model = create_model(lstm_layer_size=nn1, dense_layer_size1=nn2, dense_layer_size2=nn3,
                             grad_clip_norm=grad_clip_norm, lr=lr, l1=l1, l2=l2, output_type=output_type,
                             embeddings=embeddings, max_seq_length=max_seq_length)
        trained_model = train_model(model, x_train, y_train, x_validation, y_validation,
                                    model_name='hyper_paramter_tuning', save_model=False, batch_size=batch_size,
                                    n_epoch=5)
        val_loss, val_acc = model.evaluate((x_validation[x_labels[0]], x_validation[x_labels[1]]), y_validation,
                                           verbose=2)
        # if the validation accuracy is higher than the current best accuracy, save parameter index values
        if (best_acc < val_acc):
            best_acc = val_acc
            selected_parameters = [r1, r2, r3, r4, r5, r6, r7, r8, r9]
            print(selected_parameters)

    return selected_parameters


# ======================================================================================================================
# Train NN Models

def create_train_evaluate_model(features, x_train, y_train, x_validation, y_validation, sim_meaure, embeddings,
                                max_seq_length, model_name='',
                                save_model=False, n_epoch=25):
    """
    build, train and evaluate model

    Parameters
    ----------
    features : 'best parameters' from hyper tuning
    x_train : training data inputs
    y_train : training data labels
    x_validation : validation data inputs
    y_validation : validation data labels
    sim_meaure : specify similarity measure used
    embeddings : embedding matrix used
    max_seq_length : maximum length of questions
    model_name : set model name
    save_model : define whether to save model or not
    n_epoch : max number of training epochs
    """
    # features
    lr = lr_array[features[0]]
    decay = decay_array[features[1]]
    grad_clip_norm = grad_clip_norm_array[features[2]]
    nn1 = nn1_array[features[3]]
    nn2 = nn2_array[features[4]]
    nn3 = nn3_array[features[5]]
    l1 = l1_array[features[6]]
    l2 = l2_array[features[7]]
    batch_size = batch_array[features[8]]
    # create model
    model = create_model(lstm_layer_size=nn1, dense_layer_size1=nn2, dense_layer_size2=nn3,
                         grad_clip_norm=grad_clip_norm, lr=lr, l1=l1, l2=l2, output_type=sim_meaure,
                         embeddings=embeddings, max_seq_length=max_seq_length)
    # train model
    trained_model = train_model(model, x_train, y_train, x_validation, y_validation, model_name=model_name,
                                save_model=save_model, batch_size=batch_size, n_epoch=n_epoch)
    # evaluate model
    val_loss, val_acc = model.evaluate((x_validation[x_labels[0]], x_validation[x_labels[1]]), y_validation, verbose=2)
    return model, trained_model, val_loss, val_acc


def plot_training_accuracy(trained_model, save, name=''):
    """
    plot training-validation-accuracy

    Parameters
    ----------
    trained_model : trained model
    save : specify whether to save or not
    name : name of figure
    """
    # Plot accuracy
    plt.plot(trained_model.history['accuracy'])
    plt.plot(trained_model.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    if (save):
        plt.savefig(path + 'val_vs_train_acc_{}.png'.format(name), format='png')
    plt.show()

    # Plot loss
    plt.plot(trained_model.history['loss'])
    plt.plot(trained_model.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    if (save):
        plt.savefig(path + 'val_vs_train_loss_{}.png'.format(name), format='png')
    plt.show()


# ======================================================================================================================
# Hyper-parameter Tuning , Training and Testing for SVM model

def svc_param_selection(X, y, nfolds):
    """
    tune hyper-parameters of SVM model

    Parameters
    ----------
    X : inputed data used for tuning
    Y : labeled data used for tuning
    nfolds : number of cross-validation folds used

    """
    print('Tuning SVM parameters:')
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 'scale']
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    mean_CV_score = grid_search.cv_results_['mean_test_score']
    return best_params, mean_CV_score


def train_SVM_model(x_tr, y_tr, best_params, model_name):
    """
    tune hyper-parameters of DL model

    Parameters
    ----------
    x_tr : training data inputs
    y_tr : training data labels
    best_params : best parameters from tuning
    model_name : models name
    """
    print('Training SVM model:')
    clf = svm.SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], probability=True)
    clf.fit(x_tr[svm_features], y_tr)
    filename = path + model_name + '.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return clf


# ======================================================================================================================
# Test non-ensemble models


def test_DL_model(model, x, y):
    """
    testing DL model

    Parameters
    ----------
    model : DL model to be tested
    x : testing data inputs
    y: testing data labels

    """
    print('Testing model:')
    loss, acc = model.evaluate([x[x_labels[0]], x[x_labels[1]]], y)
    predicted = model.predict([x[x_labels[0]], x[x_labels[1]]])
    print('Test accuracy: ' + str(acc))
    print('Test loss: ' + str(loss))
    return loss, acc, predicted


def test_SVM_model(model, x, y):
    """
    testing SVM model

    Parameters
    ----------
    model : SVM model to be tested
    x : testing data inputs
    y: testing data labels

    """
    print('Testing model:')
    y_pred_prob_test = model.predict_proba(x[svm_features])
    y_pred_act_test = model.predict(x[svm_features])
    acc_test = accuracy_score(y, y_pred_act_test)
    print('Test accuracy:' + str(acc_test))
    return y_pred_prob_test, acc_test


# ======================================================================================================================
# Create Tune, and Test Ensemble Model

def ensemble_prediction(svm_pred, nn_pred, weight=0.5):
    """
    ensemble predictor

    Parameters
    ----------
    svm_pred : predictions made by svm model
    nn_pred : predictions made by dl model
    weight :  weight used to combine svm predictions and dl predictions

    """
    final_pred = weight * svm_pred + (1 - weight) * nn_pred
    return final_pred


def tune_ensemble_weight(y_val, y_pred_prob_val_svm, y_pred_prob_val_nn):
    """
    tuning ensemble weight

    Parameters
    ----------
    y_val : validation data label
    y_pred_prob_val_svm : predicted labels based on validation data from SVM model
    y_pred_prob_val_nn :  predicted labels based on validation data from DL model

    """
    best_acc = -1
    best_weight = 0.5
    for i in range(0, 11):
        w = 0.1 * i
        print('Weight:' + str(w))
        final_pred_val = ensemble_prediction(y_pred_prob_val_svm, y_pred_prob_val_nn, weight=w)
        y_final_labels_val = np.argmax(final_pred_val, axis=1)
        acc_final_val = accuracy_score(y_val, y_final_labels_val)
        print('Accuracy:' + str(acc_final_val))

        if (best_acc < acc_final_val):
            best_acc = acc_final_val
            best_weight = w

    return best_acc, best_weight


def test_ensemble_model(y_test, y_pred_prob_test, y_pred_nn_test, weight):
    """
    testing ensemble model

    Parameters
    ----------
    y_test : testing data labels
    y_pred_prob_test: predicted labels based on test data from SVM model
    y_pred_nn_test : predicted labels based on test data from DL model
    weight : best weight found during tuning

    """
    print('Testing model:')
    final_pred = ensemble_prediction(y_pred_prob_test, y_pred_nn_test, weight=weight)
    y_final_labels = np.argmax(final_pred, axis=1)
    acc_final = accuracy_score(y_test, y_final_labels)
    print('Test accuracy:' + str(acc_final))
    return acc_final
