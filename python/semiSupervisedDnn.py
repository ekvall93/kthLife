import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

""" from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.metrics import f1_score
from keras.layers.normalization import BatchNormalization """

""" def get_dnn():
    # import BatchNormalization


    # instantiate model
    model = Sequential()

    # we can think of this chunk as the input layer
    model.add(Dense(500, input_dim=500, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    # we can think of this chunk as the hidden layer    
    model.add(Dense(200, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    # we can think of this chunk as the output layer
    model.add(Dense(20, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    # we can think of this chunk as the output layer
    model.add(Dense(1, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))


    # setting up the optimization of our weights 
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd,
                    metrics=['accuracy'])
    return model """

def get_dnn():
    dnn = Sequential()
    dnn.add(Dense(500, input_dim=500, activation='sigmoid'))
    dnn.add(Dropout(0.2))
    dnn.add(Dense(200, activation='sigmoid'))
    dnn.add(Dropout(0.2))
    dnn.add(Dense(50, activation='sigmoid'))
    dnn.add(Dropout(0.2))
    dnn.add(Dense(1, activation='sigmoid'))

    dnn.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
 
    
    return dnn

""" def get_dnn():
    dnn = Sequential()
    dnn.add(Dense(500, input_dim=500, activation='relu'))
    dnn.add(Dropout(0.5))
    dnn.add(Dense(1000, activation='relu'))
    dnn.add(Dropout(0.5))
    dnn.add(Dense(500, activation='relu'))
    dnn.add(Dropout(0.5))
    dnn.add(Dense(200, activation='relu'))
    dnn.add(Dropout(0.5))
    dnn.add(Dense(50, activation='relu'))
    dnn.add(Dropout(0.5))
    dnn.add(Dense(1, activation='sigmoid'))

    dnn.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    
    return dnn """

class selfTrainer:
    def __init__(self, topk=100, save_error=False
                 ,verbose=False, specifiedModel=None,
                 epoch=20, batch_size=128, keras=True):
        self.topk = topk
        self.save_error = save_error
        self.verbose = verbose
        self.epoch = epoch
        self.batch_size = batch_size
        self.keras = keras
        if keras:
            self.specifiedModel = get_dnn()
        else:
            self.specifiedModel = GaussianNB()
            

    def _init_train(self, X_train, y_train):
        self.X_init = X_train.copy()
        self.y_init = y_train.copy().reshape(-1,1)
    def _init_unlabeled(self, X_unlabeled):
        self.X_un_init = X_unlabeled.copy()
    
    def _init_model(self, X, y):
        self.model = self.specifiedModel
        if self.keras:
            self.model.fit(X, y.ravel(), epochs=self.epoch,
                        batch_size=self.batch_size, verbose=0)
        else:
            self.model.fit(X, y.ravel())
    
    def _fit_model(self, X, y):
        if self.keras:
            self.model.fit(X, y.ravel(), epochs=self.epoch,
                           batch_size=self.batch_size, verbose=0)
        else:
            self.model.fit(X, y.ravel())

    def _get_acc_score(self, X, y, tmp=False):
        pred = np.around(self.model.predict(X))
        return accuracy_score(pred, y.ravel())

    def _get_f1_score(self, X, y, tmp=False):
        pred = np.around(self.model.predict(X))
        return f1_score(pred, y.ravel())
    
    def _get_probs(self, X_unlabeled):
        prob, labels = self.basemodel.predict_proba(X_unlabeled), self.basemodel.predict(X_unlabeled)
        return np.sort(prob, axis=1)[:, 1], labels
        
    def _init_save_error(self, test_error=False):
        if test_error:
            self.test_error_list = list()
        else:
            self.val_error_list = list()
            
    def _add_error(self, error, test_error=False):
        if test_error:
            self.test_error_list.append(error)
        else:
            self.val_error_list.append(error)
    
    def _error_dif(self, tmp=False):
        if tmp:
            return self.test_error_list[-1] - self.test_error_list[-2]
        else:
            return self.val_error_list[-1] - self.val_error_list[-2]
    def _save_model(self):
        if self.keras:
            self.model.save_weights('assets/kerasmodels/model_weights.h5')
        else:
            pass
    def _load_model(self):
        if self.keras:
            self.model.load_weights('assets/kerasmodels/model_weights.h5')
        else:
            self.model = GaussianNB()
            
    ''' def _if_better_model(self):
        return self.bestTestAcc < self.AccTest '''
    def _if_better_model(self):
        return self.bestTestf1 <= self.f1Test
        #return self.bestTestAcc <= self.AccTest
        
    def _use_test_data(self):
        return self.bestTestAcc != -1
    def _init_unlabeledData_relative_ix(self, length):
        self.relative_indices = np.arange(length)
    def _init_index_to_keep(self):
        self.keep_index = list()
    def _keep_indieces(self, ix):
        self.keep_index += list(ix)
    def _get_probability_and_labels(self, X):
        p = self.model.predict(X)
        labels = np.around(p)
        p[p <= .5] = 1 - p[p <= .5]
        return p, labels
    def _get_absolute_ix(self, p):
        sortedIx = np.argsort((p.ravel()))[::-1]
        ix = sortedIx[: self.topk]
        return ix
    def _sample_new_data(self, X):
        p, labels = self._get_probability_and_labels(X)
        ix = self._get_absolute_ix(p)
        return X[ix], labels[ix], ix
    def _add_data(self, X, y, X_new_data=None, y_new_data=None, sample=False):
        y = y.reshape(-1,1)
        if sample:
            y_new_data = y_new_data.reshape(-1,1)
            return np.concatenate((X_new_data, X)), np.concatenate((y_new_data, y))
        else:

            return np.concatenate((self.X_init, X)), np.concatenate((self.y_init, y))
    

    def _update_(self, tmp_sampleX, tmp_sampley, ix):
        self._save_model(), self._keep_indieces(ix)
        if self._use_test_data():
            self.bestTestAcc = self.AccTest
            self.bestTestf1 = self.f1Test
        
        return tmp_sampleX, tmp_sampley

    def _if_train(self, i):
        return i != -1

    def _report(self, error, iteration, new_data_n):
        print("Acc at iter {}:".format(iteration), error, "and the dif: ", self._error_dif())
        if self._use_test_data():
            print("and acc: {} , and f1: {}, and n new data: {}".format(self.bestTestAcc, self.bestTestf1, new_data_n))

    def _save_error_data(self, val_err):
        self._add_error(val_err, test_error=False)
        if self._use_test_data(): self._add_error(self.bestTestAcc, test_error=True)
    
    def _reset_unlabeled_data(self):
        self.relative_indices = np.delete(self.relative_indices,
                                          np.asarray(self.keep_index),
                                          axis=0)
        self._init_index_to_keep()
        return self.X_un_init[self.relative_indices]
    def _if_no_data(self, number_of_data_points):
        return number_of_data_points == 0

    def _init_fit(self, X_train, y_train, X_unlabeled):
        (self._init_train(X_train, y_train), self._init_unlabeledData_relative_ix(X_unlabeled.shape[0]),
         self._init_index_to_keep(), self._init_unlabeled(X_unlabeled), self._init_model(X_train, y_train),
         self._save_model())
    
    def get_model(self):
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        pred = np.around(self.model.predict(X))
        return accuracy_score(pred, y.ravel())
    
    def _judge_model(self, X, y):
        self.judge = SGDClassifier(loss="log", max_iter=1000, tol=1e-3)
        self.judge.fit(X, y.ravel())
    def _judge_score(self, X, y):
        pred = np.around(self.judge.predict(X))
        return accuracy_score(pred, y.ravel())
        
        
    def fit(self, X_train, y_train, X_test, y_test, X_unlabeled, X_val, y_val):

        
        self._init_fit(X_train, y_train, X_unlabeled)

        if self.save_error:
            if np.any(X_test): self._init_save_error(test_error=True)
            self._init_save_error(test_error=False)
                
        
        if np.any(X_test):
            self.bestTestf1 = self.f1Test = self._get_f1_score(X_test, y_test)
            self.bestTestAcc = self.AccTest = self._get_acc_score(X_test, y_test)

            if self.save_error: self._add_error(self.bestTestAcc, test_error=True)
        else:
            self.bestTestAcc, self.AccTest = -1, -1

        bestValAcc = self._get_acc_score(X_val, y_val)
        
        
        if self.verbose: print("Init val error: ", bestValAcc)
            
        if self.save_error: self._add_error(bestValAcc, test_error=False)
    
        i = 0
        sample = False
        while self._if_train(i):

            newX, newy, ix = self._sample_new_data(X_unlabeled)
            
            if not sample:
                sampleX, sampley, tmp_sampleX, tmp_sampley = None, None, newX, newy
            else:
                tmp_sampleX, tmp_sampley = self._add_data(sampleX, sampley, newX, newy, sample=sample)
        
            X_train, y_train = self._add_data(tmp_sampleX, tmp_sampley)

            self._fit_model(X_train, y_train)

            

            if self._use_test_data():

                self.AccTest = self._get_acc_score(X_test, y_test, tmp=True)
                self.f1Test = self._get_f1_score(X_test, y_test, tmp=True)


            """ if self.verbose:
                print("-------(", i, ")","Best f1: ", np.round(self.bestTestf1, 3),"New f1: ", np.round(self.f1Test, 3), X_train.shape[0], "--------")                
                print("Acc:",self._get_acc_score(X_test, y_test, tmp=True))
 """
            if self._if_better_model():
                sample = True
                sampleX, sampley = self._update_(tmp_sampleX, tmp_sampley, ix)
            else:
                if self.keras:
                    self._load_model()

            X_unlabeled = np.delete(X_unlabeled, ix, axis=0)

            if i % 1000 == 0:
                val_err = self._get_acc_score(X_val, y_val)
                self._save_error_data(val_err)
                if sample:
                    n_data = sampleX.shape[0]
                else:
                    n_data = 0
                if self.verbose: self._report(val_err, i, n_data)
                
                
                
                if self.keras:
                    self._load_model()
                    self.model.save_weights('assets/kerasmodels/model_weights_iter{}_val{}_test{}.h5'.format(i,
                                                                                                         np.round(val_err, 3),
                                                                                                         np.round(self.AccTest, 3)))

                np.save("assets/SemiSupArray/sampleX_iter{}_val{}_test{}".format(i,
                                                                               np.round(val_err, 3),
                                                                               np.round(self.AccTest, 3)),
                                                                               sampleX)

                np.save("assets/SemiSupArray/sampley_iter{}_val{}_test{}".format(i,
                                                                               np.round(val_err, 3),
                                                                               np.round(self.AccTest, 3)),
                                                                               sampley)
                
            i += 1

            if self._if_no_data(X_unlabeled.shape[0]):
                if self.verbose: print("One epoch of unlabeled data have passed; #new datapoints: ", len(self.keep_index))

                if self._if_no_data(len(self.keep_index)):
                    i = -1
                else:
                    X_unlabeled = self._reset_unlabeled_data()
                    if self._if_no_data(X_unlabeled.shape[0]):
                        i = -1
        print("Learning finnished")
            
     
