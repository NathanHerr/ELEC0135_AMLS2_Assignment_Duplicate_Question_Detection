from distutils.util import strtobool
from sys import argv
from ast import literal_eval

# ======================================================================================================================
# Main
# parse the command line argument and run corresponding code
if __name__ == '__main__':
    # Check to see if a parameter is passed with main
    if len(argv) != 2:
        print('usage: python main.py False')
        exit(-1)
    else:
        from duplicate_question_detection import *
        from duplicate_question_detection import path

    # if  "y", "yes", "t", "true", "on", and "1" passed, then do all steps from scratch
    if (bool(strtobool(argv[1]))):
        path = './from_scratch/'
        print('All processing will be performed from scratch, this will take a while.')
        quora_questions_df = load_data(path + "train.csv")
        quora_questions_df = remove_class_imbalance(quora_questions_df, 1)
        quora_questions_df = sample_data(quora_questions_df, 0.002)
        quora_questions_df = remove_nan_rows(quora_questions_df)
        normalised_questions_df = normalise_dataframe(quora_questions_df)
        sampled_normalised_questions_df = sample_data(normalised_questions_df, 0.3)
        normalised_questions_df.to_csv(path + 'test_normalised_questions_df.csv', index=False, encoding='utf-8')
        sampled_normalised_questions_df.to_csv(path + 'test_sampled_normalised_questions_df.csv', index=False,
                                               encoding='utf-8')

        sampled_normalised_questions_df = remove_nan_rows(sampled_normalised_questions_df)
        print(sampled_normalised_questions_df)
        constituent_features = get_constituent_features(sampled_normalised_questions_df)
        with open(path + "test_constituent_features.txt", "wb") as fp:  # Pickling
            pickle.dump(constituent_features, fp)

        normalised_questions_df = remove_nan_rows(normalised_questions_df)
        word2vec = load_w2v_model()
        vocabulary, inverse_vocabulary, normalised_questions_df = create_vocabulary(normalised_questions_df, word2vec)
        normalised_questions_df.to_csv(path + 'test_net_ready_normalised_questions_df.csv', index=False,
                                       encoding='utf-8')
        with open(path + "test_vocabulary.txt", "wb") as fp:  # Pickling
            pickle.dump(vocabulary, fp)
        with open(path + "test_inverse_vocabulary.txt", "wb") as fp:  # Pickling
            pickle.dump(inverse_vocabulary, fp)

        sampled_vocabulary, sampled_inverse_vocabulary, sampled_normalised_questions_df = create_vocabulary(
            sampled_normalised_questions_df, word2vec)
        sampled_normalised_questions_df.to_csv(path + 'test_net_ready_sampled_normalised_questions_df.csv', index=False,
                                               encoding='utf-8')
        with open(path + "test_sampled_vocabulary.txt", "wb") as fp:  # Pickling
            pickle.dump(sampled_vocabulary, fp)
        with open(path + "test_sampled_inverse_vocabulary.txt", "wb") as fp:  # Pickling
            pickle.dump(sampled_inverse_vocabulary, fp)

        embeddings = create_embedding_matrix(vocab=vocabulary, dim=embedding_dim, word2vec=word2vec)
        np.save(path + 'test_embeddings.npy', embeddings)
        sampled_embeddings = create_embedding_matrix(vocab=sampled_vocabulary, dim=embedding_dim, word2vec=word2vec)
        np.save(path + 'test_sampled_embeddings.npy', sampled_embeddings)

        delete_w2v_model(word2vec)  # Since word2vec model is so large, it is deleted once finished being used

        max_seq_length = max(normalised_questions_df.question1.map(lambda x: len(x)).max(),
                             normalised_questions_df.question2.map(lambda x: len(x)).max())
        sampled_max_seq_length = max(sampled_normalised_questions_df.question1.map(lambda x: len(x)).max(),
                                     sampled_normalised_questions_df.question2.map(lambda x: len(x)).max())

        print(normalised_questions_df)
        X_train, X_validation, X_test, Y_train, Y_validation, Y_test = train_val_test_split(normalised_questions_df,
                                                                                            max_seq_length)

        sampled_normalised_questions_df = remove_nan_rows(sampled_normalised_questions_df)
        constituent_features_df = pd.DataFrame(constituent_features)

        full_train_NN, full_train_SVM, train_NN, train_SVM, validation_NN, validation_SVM, test_NN, test_SVM = ensemble_train_val_test_split(
            constituent_features_df, sampled_normalised_questions_df)
        X_train_NN, X_validation_NN, X_test_NN, Y_train_NN, Y_validation_NN, Y_test_NN = reformat_train_val_test(
            train_NN, validation_NN, test_NN, 'NN', sampled_max_seq_length)
        X_train_SVM, X_validation_SVM, X_test_SVM, Y_train_SVM, Y_validation_SVM, Y_test_SVM = reformat_train_val_test(
            train_SVM, validation_SVM, test_SVM, 'SVM', sampled_max_seq_length)

        best_parameters_Manhattan = hyper_parameter_tuning(X_train, Y_train, X_validation, Y_validation, 15, 'dist',
                                                           embeddings, max_seq_length)
        best_parameters_SN = hyper_parameter_tuning(X_train, Y_train, X_validation, Y_validation, 15, 'NN', embeddings,
                                                    max_seq_length)

        print('Training Si-Bi-FFNLSTM')
        model_NN, nn_trained_model, nn_val_loss, nn_val_acc = create_train_evaluate_model(best_parameters_SN, X_train,
                                                                                          Y_train,
                                                                                          X_validation, Y_validation,
                                                                                          sim_meaure='NN',
                                                                                          embeddings=embeddings,
                                                                                          max_seq_length=max_seq_length,
                                                                                          model_name='test_Si-Bi-FFNLSTM',
                                                                                          save_model=True, n_epoch=25)

        print('Training Si-Bi-MaLSTM')
        model_DIST, dist_trained_model, dist_val_loss, dist_val_acc = create_train_evaluate_model(
            best_parameters_Manhattan,
            X_train,
            Y_train, X_validation,
            Y_validation,
            sim_meaure='dist',
            embeddings=embeddings,
            max_seq_length=max_seq_length,
            model_name='test_Si-Bi-MaLSTM',
            save_model=True, n_epoch=25)

        print('Training Si-Bi-FFNLSTM for Ensemble')
        model_NN_ensemble, nn_ensemble_trained_model, nn_ensemble_val_loss, nn_ensemble_val_acc = create_train_evaluate_model(
            best_parameters_SN, X_train_NN, Y_train_NN, X_validation_NN, Y_validation_NN,
            sim_meaure='NN',
            embeddings=sampled_embeddings,
            max_seq_length=sampled_max_seq_length, model_name='test_Si-Bi-FFNLSTM-ensemble', save_model=True,
            n_epoch=25)

        plot_training_accuracy(nn_trained_model, 'test_Si-Bi-FFNLSTM')
        plot_training_accuracy(dist_trained_model, 'test_Si-BiMaLSTM')
        plot_training_accuracy(nn_ensemble_trained_model, 'test_Si-Bi-FFNLSTM-Ensemble')

        print(full_train_SVM[svm_features])
        print(full_train_SVM[y_label])
        best_SVM_params, mean_CV_score = svc_param_selection(full_train_SVM[svm_features],
                                                             full_train_SVM[y_label], 5)
        model_SVM = train_SVM_model(X_train_SVM, Y_train_SVM, best_SVM_params, 'test_ensemble_svm_model')

        print('Testing Si-Bi-FFNLSTM model:')
        test_loss_nn, test_accuracy_nn, _ = test_DL_model(model_NN, X_test, Y_test)

        print('Testing Si-Bi-MaLSTM model:')
        test_loss_dist, test_accuracy_dist, _ = test_DL_model(model_DIST, X_test, Y_test)

        print('Testing Si-Bi-FFNLSTM-ensemble model:')
        test_loss_nn_ensemble, test_accuracy_nn_ensemble, y_pred_probability_nn = test_DL_model(model_NN_ensemble,
                                                                                                X_test_NN, Y_test_NN)

        print('Testing SVM ensemble mode:')
        y_pred_probability_svm, test_accuracy_ensemble_svm = test_SVM_model(model_SVM, X_test_SVM, Y_test_SVM)

        y_pred_prob_validation_svm, _ = test_SVM_model(model_SVM, X_validation_SVM, Y_validation_SVM)
        _, _, y_pred_prob_validation_nn = test_DL_model(model_NN_ensemble, X_validation_NN, Y_validation_NN)

        best_ensemble_val_acc, best_ensemble_weight = tune_ensemble_weight(Y_validation_SVM, y_pred_prob_validation_svm,
                                                                           y_pred_prob_validation_nn)

        test_accuracy_ensemble = test_ensemble_model(Y_test_SVM, y_pred_probability_svm, y_pred_probability_nn,
                                                     best_ensemble_weight)

        print(">>>>>>>>>> In Summary <<<<<<<<<<")
        print('Si-Bi-FFNLSTM:{}; Si-Bi-MaLSTM:{}; Si-Bi-FFNLSTM-Ensemble:{}; SVM-RBF-Ensemble:{}; Ensemble:{};'.format(
            test_accuracy_nn, test_accuracy_dist,
            test_accuracy_nn_ensemble, test_accuracy_ensemble_svm, test_accuracy_ensemble))
    # "n", "no", "f", "false", "off", and "0" passed, then only test pre-trained models using pre-processed data
    else:
        path = './pre_made/'
        print('Testing pre-trained models with already processing dataset.')
        # load a premade dataframe made up of questions which have been normalised
        normalised_questions_df = pd.read_csv(path + 'premade_normalised_questions_df.csv')
        # remove any nan rows which exist in the normalised dataframe
        normalised_questions_df = remove_nan_rows(df=normalised_questions_df)
        # load a premade dataframe made up of sampled questions which have been normalised
        sampled_normalised_questions_df = pd.read_csv(path + 'premade_sampled_normalised_questions_df.csv')
        # remove any nan rows which exist in the sampled normalised dataframe
        sampled_normalised_questions_df = remove_nan_rows(df=sampled_normalised_questions_df)

        # load normalised questions dataframe where the questions have been converted to their vocabulary equivalents
        net_ready_norm_train_df = pd.read_csv(path + 'net_ready_norm_train_df.csv')
        # reformat loaded data from strings to list
        net_ready_norm_train_df[x_labels[0]] = net_ready_norm_train_df[x_labels[0]].apply(lambda x: literal_eval(x))
        net_ready_norm_train_df[x_labels[1]] = net_ready_norm_train_df[x_labels[1]].apply(lambda x: literal_eval(x))

        # load normalised questions dataframe where the questions have been converted to their vocabulary equivalents
        net_ready_sampled_norm_train_df = pd.read_csv(path + 'net_ready_sampled_norm_train_df.csv')
        # reformat loaded data from strings to list
        net_ready_sampled_norm_train_df[x_labels[0]] = net_ready_sampled_norm_train_df[x_labels[0]].apply(
            lambda x: literal_eval(x))
        net_ready_sampled_norm_train_df[x_labels[1]] = net_ready_sampled_norm_train_df[x_labels[1]].apply(
            lambda x: literal_eval(x))

        # load premade vocabularies, inverse vocabularies
        with open(path + "pre_vocabulary.txt", "rb") as fp:  # Unpickling
            vocabulary = pickle.load(fp)
        with open(path + "pre_inverse_vocabulary.txt", "rb") as fp:  # Unpickling
            inverse_vocabulary = pickle.load(fp)
        with open(path + "pre_sampled_vocabulary.txt", "rb") as fp:  # Unpickling
            sampled_vocabulary = pickle.load(fp)
        with open(path + "pre_sampled_inverse_vocabulary.txt", "rb") as fp:  # Unpickling
            sampled_inverse_vocabulary = pickle.load(fp)

        # load premade word embeddings
        embeddings = np.load(path + 'pre_embeddings.npy')
        sampled_embeddings = np.load(path + 'pre_sampled_embeddings.npy')

        # define max length of questions in normalised dataframe
        max_seq_length = max(net_ready_norm_train_df.question1.map(lambda x: len(x)).max(),
                             net_ready_norm_train_df.question2.map(lambda x: len(x)).max())
        # define max length of questions in sampled normalised dataframe
        sampled_max_seq_length = max(net_ready_sampled_norm_train_df.question1.map(lambda x: len(x)).max(),
                                     net_ready_sampled_norm_train_df.question2.map(lambda x: len(x)).max())

        # open premade train, validation and test input and labels for DL model
        with open(path + "train_input.txt", "rb") as fp:  # Unpickling
            X_train = pickle.load(fp)
        Y_train = np.load(path + 'train_output.npy')

        with open(path + "validation_input.txt", "rb") as fp:  # Unpickling
            X_validation = pickle.load(fp)
        Y_validation = np.load(path + 'validation_output.npy')

        with open(path + "test_input.txt", "rb") as fp:  # Unpickling
            X_test = pickle.load(fp)
        Y_test = np.load(path + 'test_output.npy')

        # load premade constituent features
        with open(path + "premade_constituent_features.txt", "rb") as fp:  # Unpickling
            constituent_features = pickle.load(fp)
        constituent_features_df = pd.DataFrame(constituent_features)

        # open premade train, validation and test input and labels for ensemble model
        train_SVM = pd.read_csv(path + 'train_SVM_df.csv')
        validation_SVM = pd.read_csv(path + 'validation_SVM_df.csv')
        test_SVM = pd.read_csv(path + 'test_SVM_df.csv')

        train_NN = pd.read_csv(path + 'train_NN_df.csv')
        train_NN[x_labels[0]] = train_NN[x_labels[0]].apply(lambda x: literal_eval(x))
        train_NN[x_labels[1]] = train_NN[x_labels[1]].apply(lambda x: literal_eval(x))

        validation_NN = pd.read_csv(path + 'validation_NN_df.csv')
        validation_NN[x_labels[0]] = validation_NN[x_labels[0]].apply(lambda x: literal_eval(x))
        validation_NN[x_labels[1]] = validation_NN[x_labels[1]].apply(lambda x: literal_eval(x))

        test_NN = pd.read_csv(path + 'test_NN_df.csv')
        test_NN[x_labels[0]] = test_NN[x_labels[0]].apply(lambda x: literal_eval(x))
        test_NN[x_labels[1]] = test_NN[x_labels[1]].apply(lambda x: literal_eval(x))

        # reformat ensemble data to their correct formats respectively
        X_train_NN, X_validation_NN, X_test_NN, Y_train_NN, Y_validation_NN, Y_test_NN = reformat_train_val_test(
            train_NN, validation_NN, test_NN, 'NN', sampled_max_seq_length)
        X_train_SVM, X_validation_SVM, X_test_SVM, Y_train_SVM, Y_validation_SVM, Y_test_SVM = reformat_train_val_test(
            train_SVM, validation_SVM, test_SVM, 'SVM', sampled_max_seq_length)

        print('Testing Si-Bi-FFNLSTM model:')
        model_NN = tf.keras.models.load_model(path + 'pretrained-Si-Bi-FFNLSTM.hdf5')
        test_loss_nn, test_accuracy_nn, _ = test_DL_model(model_NN, X_test, Y_test)

        print('Testing Si-Bi-MaLSTM model:')
        model_DIST = tf.keras.models.load_model(path + 'pretrained-Si-Bi-MaLSTM.hdf5')
        test_loss_dist, test_accuracy_dist, _ = test_DL_model(model_DIST, X_test, Y_test)

        print('Testing Si-Bi-FFNLSTM-ensemble model:')
        model_NN_ensemble = tf.keras.models.load_model(path + 'pretrained-Si-Bi-FFNLSTM_ensemble.hdf5')
        test_loss_nn_ensemble, test_accuracy_nn_ensemble, nn_ensemble_predicted = test_DL_model(model_NN_ensemble,
                                                                                                X_test_NN, Y_test_NN)

        print('Testing SVM ensemble model:')
        model_SVM = pickle.load(open(path + 'pretrained_ensemble_svm_model.sav', 'rb'))
        y_pred_probability_svm, test_accuracy_ensemble_svm = test_SVM_model(model_SVM, X_test_SVM, Y_test_SVM)

        print('Testing Ensemble model')
        test_accuracy_ensemble = test_ensemble_model(Y_test_SVM, y_pred_probability_svm, nn_ensemble_predicted, 0.6)

        print(">>>>>>>>>> In Summary <<<<<<<<<<")
        print('Si-Bi-FFNLSTM:{}; Si-Bi-MaLSTM:{}; Si-Bi-FFNLSTM-Ensemble:{}; SVM-RBF-Ensemble:{}; Ensemble:{};'.format(
            test_accuracy_nn, test_accuracy_dist,
            test_accuracy_nn_ensemble, test_accuracy_ensemble_svm, test_accuracy_ensemble))
