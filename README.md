# ELEC0135_AMLS2_Assignment_Duplicate_Question_Detection

## Organisation of project and Role of Each Folder/File

-> The project's code is split into two main files, main.py and duplicate_question_detecion which are responsible for the following:

	duplicate_question_detection.py: This module contains the nesessary code to = Loading Quora Dataset and Normalising Questions
										    = Create Constituent Features
										    = Creating Word Embeddings, and Train-Validation-Test Split
										    = Create Ensemble train-validation-test split
										    = Build Si-Bi-LSTM Model
										    = Hyper-parameter Tuning for Si-Bi-LSTM models
										    = Train Si-Bi-LSTM Models
										    = Hyper-parameter Tuning , Training and Testing for SVM model
										    = Test Si-Bi-MaLSTM, Si-Bi-FFNLSTM, SVM-RBF models
										    = Create, Tune, and Test Ensemble Model
	
	main.py : This module is repsonsible for running the functions in duplicate_question_detection. 
		: When calling the main.py file, the operation mode needs to be specified, Namley, 
		  whether to process, train and test all from scratch or to used pre-processed data and pre-trained modules to test. 
		  To run everything from scratch, operation mode 'true' is used. To only test pre-trained models, operation mode 'false' is used.

 
-> The Dataset folder contains the original datasets provided as well as the final datasets
   (after performing pre-processing etc.) which are used as the final input to the system.

-> Folder 'from_scratch' is used to store all new data and models, as well as all files needed, if operation mode 'true' is used.

-> Folder 'pre_made' is used to store all pre-processed data and and pre-trained models for operation mode 'false'.

-> Folder 'other' is used to hold all old data and models.

## Main Packages Required For Running the Code

The following should be noted

       : Tensorflow 2.x is used

       : Python 3.7 must be used
       
       : If a package is missing - use pip, conda or any other package manager to install it

The packages:

  - import pandas

  - import numpy

  - import matplotlib

  - import re

  - import itertools

  - import time

  - import datetime

  - import random

  - import statistics

  - import gensim

  - import zipfile

  - import spacy

  - import unicodedata

  - import imblearn

  - import collections

  - import pickle

  - import nltk

  - import allennlp

  - import sklearn

  - import tensorflow

  - import keras

                     
