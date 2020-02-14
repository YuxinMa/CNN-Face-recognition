
# coding: utf-8



import cv2
import sys
import os
import time
import dlib
import random
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

from sklearn.model_selection  import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,Convolution2D, MaxPooling2D,SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.optimizers import SGD
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras import layers
from keras.utils.data_utils import get_file
from keras.models import Model



import pylab
pylab.rcParams['figure.figsize'] = (15.0, 8.0) # Show size
i=1
# Show dataset
patchPath = '/home/yuxinma/code/faceRecognition/FaceDB/database/'
for fileName in os.listdir(os.path.join(patchPath,'1')):
    if fileName.endswith('.HTM'):
        continue
    image = cv2.imread(patchPath+'1/'+fileName)
    img = image[...,::-1]
    axis = plt.subplot(3,len(os.listdir(os.path.join(patchPath,'1')))/3+1,i)
    plt.imshow(img)
    i+=1
plt.show()


# First use dlib for face detection
cascPath = "haarcascade_frontalface_default.xml"
dataPath = '/home/yuxinma/code/faceRecognition/FaceDB/database/'
patchPath = '/home/yuxinma/code/faceRecognition/FaceDB/faceData/'
faceCascade = cv2.CascadeClassifier(cascPath)
for personFolder in os.listdir(dataPath):
    personPath = os.path.join(dataPath,personFolder)
    for fileName in os.listdir(personPath):
        # Iterate through all pictures in a folder
        if fileName.endswith('.HTM'):
            continue
        image = cv2.imread(os.path.join(personPath,fileName))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Turn the picture into grayscale mode
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
        if(len(faces)==1):
            x,y,w,h=list(faces[0])
            facePatch=image[y:y+h,x:x+w]
            if(not os.path.exists(os.path.join(patchPath,personFolder))):
                os.mkdir(os.path.join(patchPath,personFolder)) 
            # Save all detected faces
            cv2.imwrite(os.path.join(patchPath+personFolder,fileName),facePatch)



i=1
# Visualize detected faces
patchPath='/home/yuxinma/code/faceRecognition/FaceDB/faceData/'
for fileName in os.listdir(os.path.join(patchPath,'1')):
    image=cv2.imread(patchPath+'1/'+fileName)
    img = image[...,::-1]
    axis=plt.subplot(1,len(os.listdir(os.path.join(patchPath,'1'))),i)
    plt.imshow(img)
    i+=1
plt.show()


# Data augmentation with ImageDataGenerator
detector = dlib.get_frontal_face_detector()
dataPath='/home/yuxinma/code/faceRecognition/FaceDB/database/'
patchPath='/home/yuxinma/code/faceRecognition/FaceDB/faceData/'

datagen = ImageDataGenerator(
        rotation_range=20,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        rescale=None,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        data_format='channels_last',
        cval=0,
        #channel_shift_range=0,
        vertical_flip=False)
for personFolder in os.listdir(dataPath):
    personPath=os.path.join(dataPath,personFolder)
    for fileName in os.listdir(personPath):
        if fileName.endswith('.HTM'):
            continue
        image = cv2.imread(os.path.join(personPath,fileName))
        dets = detector(image, 1)
        for i, d in enumerate(dets):
            print("person{} face{} Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(personFolder,fileName,i, d.left(), d.top(), d.right(), d.bottom()))
            facePatch=image[d.top():d.bottom(),d.left():d.right()]
            cv2.imwrite(os.path.join(patchPath+personFolder,fileName),facePatch)
            
            facePatch=img_to_array(facePatch)
            facePatch = facePatch.reshape((1,) + facePatch.shape)
            if(not os.path.exists(os.path.join(patchPath,personFolder))):
                os.mkdir(os.path.join(patchPath,personFolder)) 
                
            i = 0
            for batch in datagen.flow(facePatch, batch_size=1,
                                      save_to_dir=None#os.path.join(patchPath,personFolder), 
                                      #save_prefix=personFolder, 
                                      #save_format='jpeg'
                                      ):
                cv2.imwrite(os.path.join(patchPath+personFolder,fileName[:-4]+str(i)+'.JPG'),batch[0].astype(int))
                i += 1
                if i > 10:
                    break  # Otherwise the generator will exit the loop
            




i=1
# Demonstrate data augmentation results
patchPath='/home/yuxinma/code/faceRecognition/FaceDB/faceData/'
for fileName in os.listdir(os.path.join(patchPath,'1')):
    image=cv2.imread(patchPath+'1/'+fileName)
    img = image[...,::-1]
    axis=plt.subplot(6,len(os.listdir(os.path.join(patchPath,'1')))/6+1,i)
    plt.imshow(img)
    i+=1
plt.show()


# Read all face data and use PCA for dimensionality reduction
faceData=[[]]
label=[]
dataPath='/home/yuxinma/code/faceRecognition/FaceDB/faceData/'

for dataLabel in os.listdir(dataPath):
    folderPath=os.path.join(dataPath,dataLabel)
    for fileName in os.listdir(folderPath):
        image=cv2.imread(os.path.join(folderPath,fileName))
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        imageArray=np.array(cv2.resize(gray_image,(101,101))).reshape([101*101])/255
        label.append(int(dataLabel))
        if(faceData==[[]]):
            faceData = [imageArray]
            continue
        faceData = np.concatenate((faceData, [imageArray]),axis=0)
pca = PCA(n_components=0.95)
pca.fit(faceData)
#print ("explained_variance_ratio_",pca.explained_variance_ratio_)
#print ("explained_variance_",pca.explained_variance_)
print ("pca.n_components_",pca.n_components_)
label=np.array(label)
faceData=pca.transform(faceData)
train_images, test_images, train_labels, test_labels = train_test_split(faceData, label, test_size = 0.1, random_state = random.randint(0, 100)) 


# Classification using SVM
svc = SVC(gamma='auto')
svc.fit(train_images,train_labels)   # Train model


test_pred=svc.predict(test_images)   # Use test data to make predictions
print(classification_report(test_labels, test_pred))


# Classification using RF
clf = RandomForestClassifier(n_jobs=114)
clf.fit(train_images,train_labels) 


test_pred=clf.predict(test_images)  # Use test data to make predictions
print(classification_report(test_labels,test_pred))


IMAGE_SIZE = 101
images = []
labels = []
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'


MODEL_PATH='/home/yuxinma/code/faceRecognition/models.h5'
def read_path(path_name):    
    for dir in os.listdir(path_name):
        picFolderPath=os.path.join(path_name,dir)
        #print(dir+' '+str(len(os.listdir(picFolderPath))))
        for picname in os.listdir(picFolderPath):
            if(picname.endswith('JPG')):
                labels.append(int(dir))
                image = cv2.imread(os.path.join(picFolderPath,picname))
                image=cv2.resize(image,(101, 101), interpolation=cv2.INTER_CUBIC)
                images.append(image)
    return images,labels


# Read training data from the specified path
def load_dataset(path_name):
    images,labels = read_path(path_name)    
    
    # Turn all the input pictures into a four-dimensional array with the size
    # (IMAGE_NUMBERS * IMAGE_SIZE * IMAGE_SIZE * 3)
    images = np.array(images)
    print(images.shape)    
    return images, labels


class Dataset:
    def __init__(self, path_name):
        # Training set
        self.train_images = None
        self.train_labels = None
        
        # Validation set
        self.valid_images = None
        self.valid_labels = None
        
        # Test set
        self.test_images = None
        self.test_labels = None
        
        # Dataset load path
        self.path_name  = path_name
        
        # The order of dimensions in the current library
        self.input_shape = None
        
    # Load the dataset and divide the dataset according to the principle of cross-validation
    # and perform related preprocessing work
    def load(self, img_rows = 101, img_cols = 101, 
             img_channels = 3, nb_classes = 114):
        # Load dataset into memory
        images, labels = load_dataset(self.path_name)  
        train_images, test_valid_images, train_labels, test_valid_labels = train_test_split(images, labels, test_size = 0.2, random_state = random.randint(0, 100)) 
        valid_images, test_images, valid_labels, test_labels = train_test_split(test_valid_images, test_valid_labels, test_size = 0.5, random_state = random.randint(0, 100)) 
        
        #train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 0.3, random_state = random.randint(0, 100))        
        #_, test_images, _, test_labels = train_test_split(images, labels, test_size = 0.5, random_state = random.randint(0, 100))                
        
        # If the current dimension order is 'th', the order of input image data is: channels, rows, cols, otherwise: rows, cols, channels
        # This part of the code is to reorganize the training dataset according to the dimensional order required by keras
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)            
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)            
            
        # Print number of training, validation, and test sets
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid samples')
        print(test_images.shape[0], 'test samples')
    
        # Our model uses categorical_crossentropy as the loss function,
        # so we need to one-hot encode the category labels based on the number of categories nb_classes to vectorize them.
        # Here we have only two categories. After conversion, the label data becomes two-dimensional.
#            train_labels = np_utils.to_categorical(train_labels, nb_classes)                        
#            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)            
#            test_labels = np_utils.to_categorical(test_labels, nb_classes)                        
    
        # Float pixel data for normalization
        train_images = train_images.astype('float32')            
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')
        
        # Normalize to 0-1
        train_images /= 255
        valid_images /= 255
        test_images /= 255            
        lb=LabelBinarizer().fit(np.array(range(0,nb_classes)))
        
        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images  = test_images
        self.train_labels = lb.transform(train_labels)
        self.valid_labels = lb.transform(valid_labels)
        self.test_labels  = lb.transform(test_labels)


class myModel:
    def __init__(self):
        self.model = None 
        self.num_classes = 114
        self.weight_decay = 0.0005
        self.x_shape = [101,101,3]
        
    def buildXceptionModel(self,dataset,nb_classes=114):
        
        img_input = Input(shape=self.x_shape)

        # Block 1
        x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        # Block 2
        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        # Block 2 Pool
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        # Block 3
        x = Activation('relu')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        # Block 3 Pool
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        # Block 4
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # Block 5 - 12
        for i in range(8):
            residual = x
            x = Activation('relu')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)

            x = layers.add([x, residual])

        residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        # Block 13
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        # Block 13 Pool
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # Block 14
        x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Block 14 part 2
        x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Fully Connected Layer
        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, activation='softmax')(x)

        inputs = img_input

        # Create model
        self.model = Model(inputs, x)

        # Download and cache the Xception weights file
        #weights_path = get_file('xception_weights.h5', WEIGHTS_PATH, cache_subdir='models')

        # load weights
        #self.model.load_weights(weights_path)
        self.model.summary()

    # Build model
    def buildVGGModel(self, dataset, nb_classes = 114):
        # Construct an empty network model. It is a linear stack model.
        # Each neural network layer will be added sequentially.
        # The professional name is sequential model or linear stack model.
        self.model = Sequential() 
        weight_decay=self.weight_decay
        # The following code will sequentially add the layers required by the CNN network,
        # one add is one network layer.
        self.model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())

        self.model.add(MaxPooling2D(pool_size=(2, 2)))


        self.model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())

        self.model.add(MaxPooling2D(pool_size=(2, 2)))


        self.model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        # Output model summary
        self.model.summary()

    # Train model
    def train(self, dataset, batch_size = 64, nb_epoch = 300, data_augmentation = False):
        # Using SGD + momentum's optimizer for training, first generate an optimizer object
        sgd = SGD(lr = 0.01, decay = 1e-6,
                  momentum = 0.9, nesterov = True)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])   # Complete the model configuration work
        
        # Not use data augmentation,
        # the so-called augmentation is to use the rotation, flip, noise and other methods
        # to create new training data from the training data provided,
        # consciously increase the size of training data, increase the amount of model training
        if not data_augmentation:            
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size = batch_size,
                           nb_epoch = nb_epoch,
                           validation_data = (dataset.valid_images, dataset.valid_labels))
        # Data augmentation with real-time data
        else:            
            # Define a data generator for data augmentation.
            # It returns a generator object datagen. Each time datagen is called,
            # it generates a set of data (sequential generation) to save memory.
            # In fact, it is a Python data generator.
            datagen = ImageDataGenerator(
                featurewise_center = False,             # Whether to decentralize the input data (mean 0)
                samplewise_center  = False,             # Whether to make each sample of the input data mean 0
                featurewise_std_normalization = False,  # Whether the data is normalized (input data divided by the standard deviation of the dataset)
                samplewise_std_normalization  = False,  # Whether to divide each sample data by its own standard deviation
                zca_whitening = False,                  # Whether to apply ZCA whitening to the input data
                rotation_range = 20,                    # Random rotation angle of the picture when the data is augmented (range 0-180)
                width_shift_range  = 0.2,               # The magnitude of the horizontal shift of the picture when the data is augmented (the unit is the proportion of the picture width, floating point between 0 and 1)
                height_shift_range = 0.2,               # Same as above, but here is vertical
                horizontal_flip = True,                 # Whether to perform random horizontal flips
                vertical_flip = False)                  # Whether to perform random vertical flips

            # Calculate the number of entire training sample sets for eigenvalue normalization, ZCA whitening, etc.
            datagen.fit(dataset.train_images)                        

            # Start training the model with the generator
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                   batch_size = batch_size),
                                     samples_per_epoch = dataset.train_images.shape[0],
                                     nb_epoch = nb_epoch,
                                     validation_data = (dataset.valid_images, dataset.valid_labels))
 
    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
    def save_model(self, file_path = MODEL_PATH):
        self.model.save(file_path)
    def load_model(self, file_path = MODEL_PATH):
        self.model = load_model(file_path)


dataset = Dataset('/home/yuxinma/code/faceRecognition/FaceDB/faceData/')
dataset.load()


len(set(labels))


vggModel = myModel()


vggModel.buildVGGModel(dataset)


vggModel.train(dataset)


vggModel.save_model('/home/yuxinma/code/faceRecognition/VGGModel.h5')
vggModel.evaluate(dataset)


XceptionModel= myModel()
XceptionModel.buildXceptionModel(dataset)


XceptionModel.train(dataset)  


XceptionModel.save_model('/home/yuxinma/code/faceRecognition/XceptionModels.h5') # Save model
XceptionModel.evaluate(dataset) # Evaluate the results





