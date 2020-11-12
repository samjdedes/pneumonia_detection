import tensorflow
from tensorflow import keras

# Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

# Keras-Tuner
from kerastuner.tuners import RandomSearch

CONV_2D = 'Conv2D'
BATCH_NORMAL = 'BatchNormal'
MAXPOOL_2D = 'MaxPool2D'
DROPOUT_10PERCENT = 'DropOut_10percent'
DROPOUT_20PERCENT = 'DropOut_20percent'
FLATTEN_LAYERS = 'Flatten_layers'

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
    def __init__(self):              
        '''
            Set default parameters for calling CNN keras models
        '''
        self.model_params = {
                 'units': 32,
                 'filter': (3,3),
                 'kernel_size': 3,
                 'activation': 'relu',
                 'padding': 'same',
                 'pool_size': (2,2),
                 'strides': 1,
                 'dense_units': 2,
                 'activation':'softmax',
                 'drop_out_10percent': 0.1,
                 'drop_out_20percent': 0.2,
                 'input_shape': (150, 150, 1),
                 'optimizer': 'adam',
                 'loss': 'binary_cross_entropy',
                 'metrics': ['accuracy'],
                 'output_activation': 'sigmoid', 
                 'batch_size':32
               } 
    
    def build_model(self, layers):
        self.model = Sequential()
        for layer in layers:
            if layer == CONV_2D:
                self.model.add(Conv2D(self.model_params['units'], 
                                 self.model_params['filter'],
                                 padding = self.model_params['padding'],
                                 strides = self.model_params['strides'],
                                 activation = self.model_params['activation'],
                                 input_shape = self.model_params['input_shape']))                
            elif layer == BATCH_NORMAL:
                self.model.add(BatchNormalization())
            elif layer == MAXPOOL_2D:
                self.model.add(MaxPool2D(self.model_params['pool_size'] , 
                                    self.model_params['strides'],
                                    self.model_params['padding']))
            elif layer == DROPOUT_10PERCENT:
                self.model.add(Dropout(self.model_params['drop_out_10percent']))
            elif layer ==FLATTEN_LAYERS:
                self.model.add(Flatten())
            else: 
                if layer == DROPOUT_20PERCENT:
                    self.model.add(Dropout(model.params['drop_out_20percent']))
                
        # output layer
        self.model.add(Dense(units=1 , activation=self.model_params['output_activation']))  
        
        self.model.compile(optimizer = "adam" , loss = 'binary_crossentropy' , metrics = self.model_params['metrics'])
        
        self.model.summary()
        return self.model      
    
    def fit_model (self, X_train, y_train):
        
        pass
            
    def analyze_model (self):
        pass
    
    def test_model(self):
        pass
            
    def plot_confusion_matrix(self):
        pass
            
    def sample_pneumonia(self):
        pass
            
    def sample_normal(self):
        pass