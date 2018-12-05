# -*- coding: utf-8 -*-
"""
Functions for Machine Learning for bachelor project

Created on Tue Sep  4 10:31:34 2018

@author: JakobLab
"""
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime as datetime
from sklearn import decomposition
from sklearn.model_selection import KFold
import ProgBar
import sys
import dill


save_path = ''

def my_loss(y_true, y_pred):
    r"""Homemade weighted loss function
    
    Uses the standard Mean Absolute Error (MAE) error function, 
    but weighs it based on the distance from 0 the true value has
        
    Formula
    -------
    .. math:: \overline{|{y_{_{true}} - y_{_{pred}}}|} \cdot 2 
              - \frac{y_{_{true}}
              - y_{_{true_{min}}}}{y_{_{true_{max}}} - y_{_{true_{min}}}}  
              
    where 
    
    .. math:: y_{_{true}} = \text{array of true values}
    
    .. math:: y_{_{pred}} = \text{array of predictions from keras}
    
    .. math:: y_{_{true_{min}}} = \text{minimum value in true values}
         
    .. math:: y_{_{true_{max}}} = \text{maximum value in true values}
    
                                                    
    Parameters
    ----------
    y_true : array
        The ground truth as provided by Keras.
    y_pred : array
        The predictions as provided by Keras.
        
    Returns
    -------
    weighted_loss : array
        The weighted loss.
    """
    absErr = K.mean(K.abs(y_pred - y_true), axis=-1)
    
    num = y_true-K.min(y_true,axis=0,keepdims=True)
    den = K.max(y_true,axis=0,keepdims=True)-K.min(y_true,axis=0,keepdims=True)
    factor = 2-(num/den)
    weighted_loss = absErr*K.mean(factor)
    return weighted_loss

class Callbacks(keras.callbacks.Callback):
    """Callbacks class
    
    Class to handle different callbacks from keras
    
    Attributes
    ----------
    save_path : str
        Path to where things are saved
    """
    def on_train_begin(self, logs={}):
        """On train begin
        
        Writes the start of the training history
        """
        global save_path
        with open(os.getcwd().replace('\\','/')+save_path.replace('.',
                  ':').replace(':','-')+'/readme.md',mode='a') as fh:
            fh.write('### Training history ### \n')
            fh.write('![history.png](history.png "History")')
            fh.write('| Epoch | Loss | Val Loss | MAE | Val MAE |\n')
            fh.write('|:--------:|:--------:|:--------:|:--------:'
                     +'|:--------:|\n')
        sys.stdout.write('\r| Epoch |   Loss   | Val_Loss |    MAE   |  Val '
                         +'MAE |   MAPE   | Val MAPE | improve | time |\n')
        sys.stdout.write('\r|-------|----------|----------|----------|------'
                         +'----|----------|----------|---------|------|\n')
        self.prev_loss = 0

        return
        

    def on_train_end(self, logs={}):
        """On train end
        
        Saves the last epoch and writes the end of training history
        """
        with open(os.getcwd().replace('\\','/')+save_path.replace('.',
                  ':').replace(':','-')+'/readme.md',mode='a') as fh:
            fh.write('___\n<br/>\n\n')
        self.model.save(os.getcwd().replace('\\','/')+save_path.replace('.'
                        ,':').replace(':','-')+'/end_of_training.hdf5')
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.target = self.params['samples']
        self.progbar = ProgBar.Progbar(target = self.target,
                                       newline_on_end = False)
        self.seen = 0
        self.progbar.last_run=0
        return

    def on_epoch_end(self, epoch, logs={}):
        """On epoch end
        
        Writes the loss and metrics to the readmefile
        """
        self.progbar.update(self.seen)        
        improvement=((logs.get('val_loss')-self.prev_loss)/logs.get('val_loss'))
        with open(os.getcwd().replace('\\','/')+save_path.replace('.',
                             ':').replace(':','-')+'/readme.md',mode='a') as fh:
            fh.write('|'+str(epoch)+'|'+str(logs.get('loss'))
                     +'|'+str(logs.get('val_loss'))+'|'
                     +str(logs.get('mean_squared_error'))+'|'
                     +str(logs.get('val_mean_squared_error'))+'|'
                     +str(self.progbar.last_run)+'|\n')
        sys.stdout.write('\r|{:^7d}|'.format(epoch)
                         +'{:^10.3g}|'.format(logs.get('loss'))
                         +'{:^10.3g}|'.format(logs.get('val_loss'))
                         +'{:^10.3g}|'.format(logs.get('mean_absolute_error'))
                         +'{:^10.3g}|'.format(logs.get('val_mean_absolute'
                                                       +'_error'))
                         +'{:^10.2g}|'.format(logs.get('mean_absolute_'
                                                       +'percentage_error'))
                         +'{:^10.2g}|'.format(logs.get('val_mean_absolute'
                                                       +'_percentage_error'))
                         +'{:^9.3f}|'.format(improvement*100)
                         +'{:^6.2f}|\n'.format(self.progbar.last_run))
                          
        self.prev_loss = logs.get('val_loss')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        self.progbar.update(self.seen)
        
        return

def Split(Data,split=0.9,seed=2018):
    """Splits the data into training and test data
    
    Splits the data using pandas functionalities
    
    
    Parameters
    ----------
    Data : pandas dataframe
        The data to split.
    split : float
        The fraction of data to keep fra training.
    seed : int
        A seed to randomize by (to get consisten results)
    
    Returns
    -------
    train_label : pandas dataframe
        The ground truth for training.
    train_dat : pandas dataframe
        The variables for training.
    test_label : pandas dataframe
        The ground truth for test.
    test_dat : pandas dataframe
        The variables for test.
        
    See Also
    --------
    pandas.df.sample : Randomly samples a fraction of the total dataframe
    pandas.df.drop : Drops part of a dataframe
    pandas.df.loc : Returns a defined part of a dataframe
    """
    train=Data.sample(frac=split,random_state=seed)
    test=Data.drop(train.index)
    
    train_lab = train.loc[:, 'Diesel':'Total']
    train_dat = train.loc[:, 'Year':]

    test_lab = test.loc[:, 'Diesel':'Total']
    test_dat = test.loc[:, 'Year':]
    
    return train_lab,train_dat,test_lab,test_dat

def Split2(Data,split=0.75,seed=1993):
    """Splits the data into training and test data
    
    Deprecated
    
    Data is the dataframe
    split is the fraction for training data
    seed is a random seed
    
    return Train label , Train data , Test label, Test data
    """
    train=Data.sample(frac=split,random_state=seed)
    test=Data.drop(train.index)

    
    return train,test

def SimpleModel(train_data, optimizer=None):
    """Creates a Artificial Neural Network model object
    
    The model is used to predict energy production from weather data.
    Sequential model with 7 hidden layers with 1024 densely connected nodes,
    with ReLU activation function.
    The model uses a custom loss function, and observes the metrics,
    Mean Absolute Error and Mean Absolut Percentage Error  
    
    
    Parameters
    ----------
    train_data : pandas dataframe
        Training data, used to shape the input.
    
    optimizer : object , optional
        Optimizer object for the model.
        default is: 
        keras.optimizers.Adam(lr=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-4)
    
    Returns
    -------
    model : object
        The compiled model used for training.
        
    See Also
    --------
    keras.optimizers.Adam : innitialize adam optimizer object
    """
    
    model = keras.Sequential([
    keras.layers.Dense(train_data.shape[1], name='Input', activation='linear',
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(1024,name='Layer1', activation='relu'),
    keras.layers.Dense(1024,name='Layer2', activation='relu'),
    keras.layers.Dense(1024,name='Layer3', activation='relu'),
    keras.layers.Dense(1024,name='Layer4', activation='relu'),
    keras.layers.Dense(1024,name='Layer5', activation='relu'),
    keras.layers.Dense(1024,name='Layer6', activation='relu'),
    keras.layers.Dense(1024,name='Layer7', activation='relu'),
    keras.layers.Dense(4,  name='Output')])

    if optimizer is None:
        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.95, beta_2=0.999,
                                          epsilon=1e-4)
        
    model.compile(loss=my_loss, optimizer=optimizer, metrics=['mae','mape'])
    model.summary()
    return model

def ModularModel(train_data, optimizer=None, layers=7, nodes=4,
                 activation='relu'):
    """Creates a Artificial Neural Network model object
    
    The model is used to predict energy production from weather data.
    The function can create a model, depending on the arguments
    The model uses a custom loss function, and observes the metrics,
    Mean Absolute Error and Mean Absolut Percentage Error  
    
    
    Parameters
    ----------
    train_data : pandas dataframe
        Training data, used to shape the input.
    
    optimizer : object , optional
        Optimizer object for the model.
        default is: 
        keras.optimizers.Adam(lr=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-4)
        
    layers : int , optional
        Number of layers
        
    nodes : int , optional
        Number of nodes in each layer 
        
    activation : str , optional
        Activation unction used on all layers
        
    Returns
    -------
    model : object
        The compiled model used for training.
        
    See Also
    --------
    keras.optimizers.Adam : innitialize adam optimizer object
    """
    model = keras.Sequential()
    model.add(keras.layers.Dense(train_data.shape[1],name='Input',
                                 activation='linear',
                                 input_shape=(train_data.shape[1],)))
    for n in range(layers):
        model.add(keras.layers.Dense(nodes,name='layer'+str(n+1),
                                     activation=activation))
    model.add(keras.layers.Dense(4,  name='Output'))

    if optimizer is None:
        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.95, beta_2=0.999,
                                          epsilon=1e-4)
        
    model.compile(loss=my_loss, optimizer=optimizer, metrics=['mae','logcosh','mape'])
    model.summary()
    return model

def TrainModel(model, train_data, train_labels, EPOCHS=500, split=0.2,
               val_data=None, checkpoint_path=None,file_name=None,
               PERIOD=5, patience=20,BATCH=32,Descriptor='',min_delta=0.05):
    """Trains the model
    
    The model is used to predict energy production from weather data.
    Creates a readme file to easily see the settings and history, 
    during training
    
    Parameters
    ----------
    model : object
        Compiled model object.
    
    train_data : dataframe
        A dataframe of the training data.
    
    train_labels : dataframe
        A dataframe of the training labels.
        
    EPOCHS : int, optional
        The number of epochs to run the training. (default = 500)
    
    split : float, optional
        The validation split. (default = 0.2)
        Only valid if val_data = None. 
        
    val_data : array-like, optional
        Validation data. (default = None)
        Overwrites split if not None.
        
    checkpoint_path : str, optional
        Path from working directory to save data. (default = None).
        if None, no checkpoints saved.
        
    file_name : str, optional
        File name to give checkpoint files. (default = None)
    
    PERIOD : int, optional
        The number of Epochs between each checkpoint. (default = 5)   
    
    patience : int, optional
        The number of epochs to check for improvement. (default = 20)
        
    BATCH : int, optional
        The batchsize to train on. (default = 32)
    
    Descriptor : str, optional
        Description of the model, for the readme file (default = '')
        
    min_delta : float
        The minimum improvement to keep training (default = 0.025)
    
    Returns
    -------
    history : object
        Object containing the history of the training.
    """
    
    if file_name is None:
        file_name='weights.Epoch-{epoch:03d};Loss-{val_loss:.6f}.hdf5'
    
    global save_path
    save_path = checkpoint_path
    callback = [Callbacks()]
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience,
                                               min_delta=min_delta)
    callback.append(early_stop)
    if checkpoint_path is not None:
        if not os.path.exists(os.getcwd().replace('\\','/')
        +checkpoint_path.replace('.',':').replace(':','-')):
            os.makedirs(os.getcwd().replace('\\','/')
            +checkpoint_path.replace('.',':').replace(':','-'))
        model_save = tf.keras.callbacks.ModelCheckpoint(
                os.getcwd().replace('\\','/')+checkpoint_path.replace('.',
                         ':').replace(':','-')+'/'+file_name, verbose=1,
                         save_weights_only=False, period=PERIOD)
    
    if PERIOD != 0:
        callback.append(model_save)
    
    #Create readme with settings and information
    with open(os.getcwd().replace('\\','/')+checkpoint_path.replace('.',
                             ':').replace(':','-')+'/readme.md',mode='w') as fh:
        fh.write('# Readme for ML model # \n')
        fh.write('*'+Descriptor+'* \n')
        fh.write('<br/>\n\n')
        fh.write('### 1 Model summary ### \n')
        fh.write('### 2 Optimizer settings ### \n')
        fh.write('### 3 Training settings ### \n')
        fh.write('###    3.1 Saving checkpoints ### \n')
        fh.write('###    3.2 Early Stopping ### \n')
        fh.write('### Training History ### \n')
        fh.write('___\n')
        fh.write('<br/>\n<br/>\n\n')
                 
        fh.write('### 1 Model Summary ### \n')
        fh.write('<br/>\n\n')
        fh.write('| Layer (type) | Output Shape | Param # |\n')
        fh.write('|:------------:|:------------:|:-------:|\n')
        for layer in model.layers:
            fh.write('| '+layer.name+' ('+layer.__class__.__name__+') | '
                     +str(layer.get_output_at(0).get_shape())+' | '
                     +str(layer.count_params())+' |\n') 
        train_param = sum([keras.backend.count_params(p) for p in set(
                                                    model.trainable_weights)])
        non_train_param = sum([keras.backend.count_params(p) for p in set(
                                                model.non_trainable_weights)])
        fh.write('\n')
        fh.write('Total parameters: '+str(train_param+non_train_param)
                 +'<br/>\n')
        fh.write('Trainable parameters: '+str(train_param)+'<br/>\n')
        fh.write('Non-trainable parameters: '+str(non_train_param)+'<br/>\n')
        
        fh.write('___\n')
        fh.write('<br/>\n<br/>\n\n')
        
        fh.write('### 2 Optimizer Settings ### \n')
        for x in model.optimizer.get_config():
            fh.write(x +' : '+str(model.optimizer.get_config()[x])+'<br/>\n')
        fh.write('___\n')
        fh.write('<br/>\n<br/>\n\n')
        
        fh.write('### 3 Training settings ### \n')
        fh.write('Epochs: '+str(EPOCHS)+'<br/>\nValidation split: '
                 +str(split)+'<br/>\n')
        fh.write('Batch Size: '+str(BATCH)+'<br/>\n')

        fh.write('#### 3.1 Saving Checkpoints #### \n')
        if checkpoint_path is None:
            fh.write('No checkpoints saved<br/>\n')
        else:
            fh.write('Checkpoints saved every '+str(PERIOD)+' epoch(s)<br/>\n')
        
        fh.write('#### 3.2 Early Stopping #### \n')
        fh.write('Early stopping active with patience of '
                 +str(patience)+',<br/>\n')
        fh.write('and a min delta of '+str(min_delta)+'<br/>\n')
        fh.write('___\n')
        fh.write('<br/>\n<br/>\n\n')
        
    # Store training stats
    history = model.fit(train_data, train_labels,epochs=EPOCHS,batch_size=BATCH,
                    validation_split=split,validation_data=val_data, verbose=0,
                    callbacks = callback)

    return history

def PlotHistory(history,save_path = None):
    """Plots the training history
    
    Plots the `loss` and `val_loss` from the history object
    
    Parameters
    ----------
    history : object
        The training history of the model.
    
    save_path : str, optional
        Path from working directory to save graphs. (default = None)
        If None, doesn't save anything
    """
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history.epoch, np.array(history.history['loss']),
             label = 'Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
             label = 'Val loss')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label = 'Train MAE')
    plt.plot(history.epoch,np.array(history.history['val_mean_absolute_error']),
             label = 'Val MAE')
    plt.legend()
    if save_path is not None:
        plt.savefig(os.getcwd().replace('\\','/')
                    +save_path.replace('.',':').replace(':','-')+'/history.png')
    plt.close()
    
def PlotHistoryfinal(history,save_path = None):
    """Plots the training history
    
    Plots the `loss` and `val_loss` from the history object
    
    Parameters
    ----------
    history : object
        The training history of the model.
    
    save_path : str, optional
        Path from working directory to save graphs. (default = None)
        If None, doesn't save anything
    """
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss [% maximum total production]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error'])/53*100,
             label = 'Train MAE')
    plt.plot(history.epoch,np.array(history.history['val_mean_absolute_error'])/53*100,
             label = 'Val MAE')
    plt.legend()
    if save_path is not None:
        plt.savefig(os.getcwd().replace('\\','/')
                    +save_path.replace('.',':').replace(':','-')+'/history.png')
    plt.close()
    



def Predict(model, test_data):
    """Predict on test_data
    
    Runs a prediction on the `model`, using the `test_data`
     
    Parameters
    ----------
    model : object
        The model to train on.
        
    test_data : array-like
        The input to predict on
        
    Returns
    -------
    test_predictions : array-like
        The predicted output values
    """
    test_predictions = model.predict(test_data)
    return test_predictions


def PCAExploration(data):
    """Principle Component Analasys - part 1
    
    Make a PC analasys, to find the most important parts, 
    of the data.
    This is the exploration part, doing a PCA using all dimensions.
    creates a graph of explained variance and prints it accumulated.
     
    Parameters
    ----------
    data : array-like
        The input data.
    """
    # Create a regular PCA model 
    pca = decomposition.PCA(n_components=data.shape[1])
    
    # Fit and transform the data to the model
    reduced_data_pca = pca.fit_transform(data)
    
    # Inspect the shape
    reduced_data_pca.shape
    
    # Print out the data
    #print(reduced_data_rpca)
    #print(reduced_data_pca)
        
    fig = plt.figure(figsize=(10,8))
    sing_vals = np.arange(data.shape[1]) + 1
    plt.plot(sing_vals, pca.explained_variance_ratio_, 'o-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Prop. explained variance')
    plt.plot(sing_vals, pca.explained_variance_ratio_.cumsum(),'o-',linewidth=2)
    plt.legend(['Individual','Cumulative','rand_Individual','rand_Cumulative'],
               loc='best', borderpad=0.3, 
                shadow=False,
                markerscale=0.4)
    
    print(pca.explained_variance_ratio_.cumsum())
    
def PCAReduction(data,test_data,comp):
    """Principle Component Analasys - part 2
    
    Make a PC analasys, to find the most important parts, 
    of the data.
    This is the reduction part, doing a PCA using the given dimensions.
    Prints the explained variance at the given dimensions
     
    Parameters
    ----------
    data : array-like
        The input data.
    
    test_data : array-like
        The test data, to keep the same components
        
    comp : int
        The number of dimensions to keep
        
    Returns
    -------
    reduced_data_pca : array-like
        The reduced training_data
    
    reduced_test_data : array-like
        The reduced test data
    """
    # Create a regular PCA model 
    pca = decomposition.PCA(n_components=comp)
    
    # Fit and transform the data to the model
    reduced_data_pca = pca.fit_transform(data)
    reduced_test_data = pca.transform(test_data)
            
    
    print('variance explained: '+str(pca.explained_variance_ratio_.cumsum()[-1]))
    return reduced_data_pca, reduced_test_data


def CrossValidation(data, labels, test, start, finish, optimizer=None,
                    layers=7, nodes=4):
    """Find the optimal structure or hyperparameter of model
    
    Runs a Cross Validation to find the optimal model to use,
    within the defined area
     
    Parameters
    ----------
    data : array-like
        The input data.
    
    labels : array-like
        The output data.
        
    test : str
        The type of test to run.
        
    start : int
        Where to start the test.
        
    finish : int
        Where to finish the test.
        
    optimizer : keras optimizer , optional
        What type of optimizer to use.
        
    layers : int , optional
        The number of layers to use (default = 7)
        
    nodes : int , optional
        The number of nodes in each layer multiplied by 256 (default = 4)
        
    Returns
    -------
    listOfErrors : dict
        A dictionary of the errors from each testrun
    """
    filefmt = 'weights.Epoch-{epoch:03d};Loss-{val_loss:.6f}.hdf5'
    #listOfErrors = list()
    kf = KFold(n_splits=10,shuffle=True,random_state=42)
    Tests = dict()
    
    for x in range(finish-start):
        x=x+start
        Iterations = dict()
        layers = layers
        nodes = nodes
        if test == 'layers':
            layers = x
            testval = layers
        elif test == 'nodes':
            nodes = x
            testval = nodes*256
        elif test == 'hyperparam':
            testval = ''
            pass
        else:
            print('Unknown test')
            return 0
        y=0
        
        if os.path.isfile(os.getcwd().replace('\\','/')
                                +'/tmp/CV/'+str(x+start+1)+'/'+str(1)+'/'
                                +'end_of_training.hdf5'):
            print(str(x+start)+' is passed')
            
        else:
            for train_index, test_index in kf.split(data):
                y += 1
                X_train, X_test = data.iloc[train_index], data.iloc[test_index]
                Y_train, Y_test=labels.iloc[train_index],labels.iloc[test_index]
                descriptor = ('CV test for '+str(test)+' layers='+str(layers)
                                +', nodes='+str(nodes)+' run: '
                                +str(x+start)+','+str(y))

                if os.path.isfile(os.getcwd().replace('\\','/')
                                    +'/tmp/CV/'+str(x+start)+'/'+str(y)+'/'
                                    +'end_of_training.hdf5'):
                    try:
                        with open(os.getcwd().replace('\\','/')
                                  +'/tmp/CV/Iterations.dill', 'rb') as f:
                            Iterations = dill.load(f)
                    except FileNotFoundError:
                        print('Iterations file not existant')
                    try:
                        with open(os.getcwd().replace('\\','/')
                                    +'/tmp/CV/Tests.dill', 'rb') as f:
                            Tests = dill.load(f)
                    except FileNotFoundError:
                        print('Tests file not existant')
                        
                    print(str(x+start)+','+str(y)+' is passed')

                else:
                    Model = ModularModel(data,optimizer=optimizer,
                                         layers = layers, nodes = nodes*256,)
                    History = TrainModel(Model, X_train, Y_train,
                               EPOCHS=500, min_delta=0.0, patience=10, PERIOD=0,
                                BATCH=45, val_data=tuple([X_test,Y_test]),
                                checkpoint_path='/tmp/CV/'+str(x+start)+'/'
                                +str(y)+'/',file_name=filefmt,
                                Descriptor=descriptor)
        
                    Iterations['run'+str(y)]={'loss':History.history['loss'][-1],
                               'val_loss': History.history['val_loss'][-1],
                               'MAE': History.history['mean_absolute_error'][-1],
                               'val_MAE': History.history[
                                       'val_mean_absolute_error'][-1],
                               'Epoch': History.epoch[-1]}
                    with open(os.getcwd().replace('\\','/')
                              +'/tmp/CV/Iterations.dill', 'wb') as f:
                        dill.dump(Iterations,f)
            
            
                    del Model
                    del History
                del X_train
                del X_test
                del Y_train
                del Y_test
            Tests[test+str(testval)] = Iterations
            with open(os.getcwd().replace('\\','/')
                      +'/tmp/CV/Tests.dill', 'wb') as f:
                    dill.dump(Tests,f)
    listOfErrors = {'info': {'testtype': test, 'layers': layers,
                             'nodes': nodes*256}, 'runs': Tests}
    return listOfErrors