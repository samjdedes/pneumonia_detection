import tensorflow
from tensorflow import keras

# Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

#tqdm
import sys
from sys import stderr
import numpy as np
import pandas as pd
import six
from keras.callbacks import Callback
# from tqdm import tqdm
# import tensorflow_addons as tfa  # =pip install -U tensorflow-addons

from sklearn.metrics import classification_report,confusion_matrix


# Keras-Tuner
from kerastuner.tuners import RandomSearch

import matplotlib.pyplot as plt
import seaborn as sns


CONV2D_3x3_32 = 'Conv2D3x3_32'
CONV2D_3x3_64 = 'Conv2D3x3_64'
CONV2D_3x3_128 = 'Conv2D3x3_128'
CONV2D_2x2_32 = 'Conv2D2x2_32'
CONV2D_2x2_64 = 'Conv2D2x2_64'
CONV2D_2x2_128 = 'Conv2D2x2_128'

BATCH_NORMAL = 'BatchNormal'
MAXPOOL_2D = 'MaxPool2D'
DROPOUT_10PERCENT = 'DropOut_10percent'
DROPOUT_20PERCENT = 'DropOut_20percent'
FLATTEN_LAYERS = 'Flatten_layers'
DENSE = 'Dense'
DENSE_512 = 'Dense512'
DENSE_128 = 'Dense128'

def augment_images(images):
    ''' 
        Purpose: Data augmentation to prevent overfitting and handling the imbalance in dataset
        Function: augment_images    
        Input: array of images
        Output: return output of .fit

    '''
    datagen = ImageDataGenerator(
                                    featurewise_center=False,  # set input mean to 0 over the dataset
                                    samplewise_center=False,  # set each sample mean to 0
                                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                    samplewise_std_normalization=False,  # divide each input by its std
                                    zca_whitening=False,  # apply ZCA whitening
                                    rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
                                    zoom_range = 0.2, # Randomly zoom image 
                                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                                    horizontal_flip = True,  # randomly flip images
                                    vertical_flip=False)  # randomly flip images )

    datagen.fit(images)
    return datagen   

    
class cnn_model:
    '''
        Takes data from the Chest X-Ray dataset and processes it for modeling.  This includes: 
            1) Set default keras deep network parameters
            2) Sets hyper-parameters  

    '''
    def __init__(self,  X_train, y_train, X_test, y_test, X_val, y_val):              
        '''
            Set default parameters for calling CNN keras models
        '''
        self.model = Sequential()
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        
        self.datagen = augment_images(self.X_train)
        
        self.model_params = {
                 'units': 32,
                 'filter': (3,3),
                 'kernel_size': 3,
                 'activation': 'relu',
                 'padding': 'same',
                 'pool_size': (2,2),
                 'strides': 2,
                 'dense_units': 2,
                 'drop_out_10percent': 0.1,
                 'drop_out_20percent': 0.2,
                 'input_shape': (150, 150, 3),
                 'optimizer': 'adam',
                 'loss': 'binary_cross_entropy',
                 'metrics': ['accuracy', 'Recall'],
                 'output_activation': 'sigmoid', 
                 'batch_size':32
               } 
    
    def build_model(self, layers):

        for layer in layers:
            if layer == CONV2D_3x3_32:  
                self.model_params['strides'] = 1
                self.model.add(Conv2D(self.model_params['units'], 
                                 self.model_params['filter'],
                                 padding = self.model_params['padding'],
                                 strides = self.model_params['strides'],
                                 activation = self.model_params['activation'],
                                 input_shape = self.model_params['input_shape']))                              
            if layer == CONV2D_3x3_64: # 64 units, 3x3 filter, 'relu', stride 2
                self.model_params['units'] = 64
                self.model_params['strides'] = 1
                self.model_params['filter'] = (3,3)
                self.model.add(Conv2D(self.model_params['units'], 
                                 self.model_params['filter'],
                                 padding = self.model_params['padding'],
                                 strides = self.model_params['strides'],
                                 activation = self.model_params['activation']))   
            if layer == CONV2D_3x3_128: # 128 units, 3x3 filter, 'relu', stride 2
                self.model_params['units'] = 128
                self.model.add(Conv2D(self.model_params['units'], 
                                 self.model_params['filter'],
                                 padding = self.model_params['padding'],
                                 strides = self.model_params['strides'],
                                 activation = self.model_params['activation'],
                                 input_shape = self.model_params['input_shape']))  
            if layer == CONV2D_2x2_32:   # default 32 units, 3x3 filter, 'relu', stride 2
                self.model_params['units'] = 32
                self.model_params['filter'] = (2,2)
                self.model.add(Conv2D(self.model_params['units'], 
                                 self.model_params['filter'],
                                 padding = self.model_params['padding'],
                                 strides = self.model_params['strides'],
                                 activation = self.model_params['activation'],
                                 input_shape = self.model_params['input_shape']))    
            if layer == CONV2D_2x2_64: # 64 units, 3x3 filter, 'relu', stride 2
                self.model_params['units'] = 64
                self.model_params['filter'] = (2,2)
                self.model_params['strides'] = 1
                self.model.add(Conv2D(self.model_params['units'], 
                                 self.model_params['filter'],
                                 padding = self.model_params['padding'],
                                 strides = self.model_params['strides'],
                                 activation = self.model_params['activation'],
                                 input_shape = self.model_params['input_shape']))   
            if layer == CONV2D_2x2_128: # 64 units, 3x3 filter, 'relu', stride 2
                self.model_params['units'] = 128
                elf.model_params['filter'] = (2,2)
                self.model.add(Conv2D(self.model_params['units'], 
                                 self.model_params['filter'],
                                 padding = self.model_params['padding'],
                                 strides = self.model_params['strides'],
                                 activation = self.model_params['activation'],
                                 input_shape = self.model_params['input_shape'])) 
            elif layer == BATCH_NORMAL:
                self.model.add(BatchNormalization())
            elif layer == MAXPOOL_2D:
                self.model_params['strides'] = 2
                self.model_params['pool_size'] = (2,2)
                self.model.add(MaxPool2D( self.model_params['pool_size'],
                                    self.model_params['strides'],
                                    self.model_params['padding']))
            elif layer == DROPOUT_10PERCENT:
                self.model.add(Dropout(self.model_params['drop_out_10percent']))
            elif layer == FLATTEN_LAYERS:
                self.model.add(Flatten())
            elif layer == DENSE_128:
                self.model.add(Dense(units=128, activation='relu'))
            elif layer == DENSE_512:
                self.model.add(Dense(units=512, activation='relu'))
            else: 
                if layer == DROPOUT_20PERCENT:
                    self.model.add(Dropout(self.model_params['drop_out_20percent']))
                
        # output layer
        self.model.add(Dense(units=1 , activation=self.model_params['output_activation']))  
        
        self.model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = self.model_params['metrics'])        
        self.model.summary()
        return self.model      
    
    def fit_report_model (self, rpt_title, BATCH_SIZE=32, EPOCHS=12):
        self.datagen_train = augment_images(self.X_train)  
        self.batch_size = BATCH_SIZE
        self.report_title = rpt_title
        self.epochs = EPOCHS
        
        # fit & report the model and  return history                                 
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=0,factor=0.3, min_lr=0.000001)
        self.history = self.model.fit(self.datagen.flow(self.X_train,self.y_train, batch_size = self.batch_size) , epochs = self.epochs, verbose=0,
                          validation_data = self.datagen.flow(self.X_val, self.y_val) ,callbacks = [learning_rate_reduction])   
        self.report_model()
        return self.history
            
    def report_model (self):
        epochs = [i for i in range(self.epochs)]
        labels = ['PNEUMONIA', 'NORMAL'] 
        history = self.history
        
        print(history.history)
        train_accuracy  = history.history['accuracy']
        train_loss = history.history['loss']
        train_recall  = history.history['recall']

        val_accuracy  = history.history['val_accuracy']
        val_loss = history.history['val_loss']

        fig , ax = plt.subplots(1,2)
        fig.set_size_inches(20,10)

        ax[0].plot(epochs , train_accuracy , 'go-' , label = 'Training Accuracy')
        ax[0].plot(epochs , val_accuracy , 'ro-' , label = 'Validation Accuracy')
        ax[0].set_title('Training & Validation Accuracy ')
        ax[0].legend()
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Accuracy")

        ax[1].plot(epochs , train_recall , 'g-o' , label = 'Training Recall')
    #   ax[1].plot(epochs , val_recall , 'r-o' , label = 'Validation Recall')
        ax[1].set_title('Traing Recall')
        ax[1].legend()
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Recall")
        plt.show()
        fig.savefig('../../visualization/{} model_accuracy_recall.png'.format(self.report_title), dpi=150)        
        self.plot_confusion_matrix()
            
    def plot_confusion_matrix(self):
        # Print Confustion Matrix

        labels = ['PNEUMONIA', 'NORMAL'] 
        predictions = self.model.predict_classes(self.X_test)
        predictions = predictions.reshape(1,-1)[0]

        print(classification_report(self.y_test, predictions, target_names = ['Pneumonia (Class 0)','Normal (Class 1)']))

        #Print Confusion matrix
        cm = confusion_matrix(self.y_test,predictions)
        print( cm)

        cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])  
        plt.figure(figsize = (10,10))
        sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='',xticklabels = labels,yticklabels = labels)
        plt.savefig('../../visualization/{} confusion-metrics.png'.format(self.report_title), dpi=150)

        self.sample_pneumonia(predictions)  
        self.sample_normal(predictions)

    def sample_pneumonia(self,predictions):
        # Print sample of Normal X-Rays   )
        print("Sample of Penumonia X-rays")
        correct = np.nonzero(predictions == self.y_test)[0]
        f = plt.figure(figsize=(16,16))
        gs = f.add_gridspec(2,3)

        with sns.axes_style("darkgrid"):
            ax = f.add_subplot(gs[0, 0])
            plt.imshow(self.X_test[0].reshape(150,150), cmap="gray", interpolation='none')
            plt.title("Predicted Class {},Actual Class {}".format(predictions[0], self.y_test[0]))

        with sns.axes_style("darkgrid"):
            ax = f.add_subplot(gs[0, 1])
            plt.imshow(self.X_test[1].reshape(150,150), cmap="gray", interpolation='none')
            plt.title("Predicted Class {},Actual Class {}".format(predictions[1], self.y_test[1]))

        with sns.axes_style("darkgrid"):
            ax = f.add_subplot(gs[0,2])
            plt.imshow(self.X_test[2].reshape(150,150), cmap="gray", interpolation='none')
            plt.title("Predicted Class {},Actual Class {}".format(predictions[2], self.y_test[2]))

        with sns.axes_style("darkgrid"):
            ax = f.add_subplot(gs[1, 0])    
            plt.imshow(self.X_test[3].reshape(150,150), cmap="gray", interpolation='none')
            plt.title("Predicted Class {},Actual Class {}".format(predictions[3], self.y_test[3]))

        with sns.axes_style("darkgrid"):
            ax = f.add_subplot(gs[1,1])
            plt.imshow(self.X_test[4].reshape(150,150), cmap="gray", interpolation='none')
            plt.title("Predicted Class {},Actual Class {}".format(predictions[4], self.y_test[4]))

        with sns.axes_style("darkgrid"):
            ax = f.add_subplot(gs[1,2]) 
            plt.imshow(self.X_test[5].reshape(150,150), cmap="gray", interpolation='none');
            plt.title("Predicted Class {},Actual Class {}".format(predictions[5], self.y_test[5]))
        f.tight_layout()
        f.savefig("../../visualization/{}Pneuominia_Validation_Images.png".format(self.report_title))
        
    def sample_normal(self, predictions):
        print("sample Normal X_rays")
        incorrect = np.nonzero(predictions != self.y_test)[0]
        f = plt.figure(figsize=(16, 16))
        gs = f.add_gridspec(2,3)

        with sns.axes_style("darkgrid"):
            ax = f.add_subplot(gs[0, 0])
            plt.imshow(self.X_test[0].reshape(150,150), cmap="gray", interpolation='none')
            plt.title("Predicted Class {},Actual Class {}".format(predictions[0], self.y_test[0]))

        with sns.axes_style("darkgrid"):
            ax = f.add_subplot(gs[0, 1])
            plt.imshow(self.X_test[1].reshape(150,150), cmap="gray", interpolation='none')
            plt.title("Predicted Class {},Actual Class {}".format(predictions[1], self.y_test[1]))

        with sns.axes_style("darkgrid"):
            ax = f.add_subplot(gs[0,2])
            plt.imshow(self.X_test[2].reshape(150,150), cmap="gray", interpolation='none')
            plt.title("Predicted Class {},Actual Class {}".format(predictions[2], self.y_test[2]))

        with sns.axes_style("darkgrid"):
            ax = f.add_subplot(gs[1, 0])    
            plt.imshow(self.X_test[3].reshape(150,150), cmap="gray", interpolation='none')
            plt.title("Predicted Class {},Actual Class {}".format(predictions[3], self.y_test[3]))

        with sns.axes_style("darkgrid"):
            ax = f.add_subplot(gs[1,1])
            plt.imshow(self.X_test[4].reshape(150,150), cmap="gray", interpolation='none')
            plt.title("Predicted Class {},Actual Class {}".format(predictions[4], self.y_test[4]))

        with sns.axes_style("darkgrid"):
            ax = f.add_subplot(gs[1,2]) 
            plt.imshow(self.X_test[5].reshape(150,150), cmap="gray", interpolation='none');
            plt.title("Predicted Class {},Actual Class {}".format(predictions[5], self.y_test[5]))
        f.tight_layout()
        f.savefig("../../visualization/{} Normal_Validation_Images.png".format(self.report_title))
        