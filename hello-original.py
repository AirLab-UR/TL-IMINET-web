import sys
import os
#os.environ['THEANO_FLAGS'] = "device=gpu0"

import theano
import keras
import numpy as np
import scipy.io
np.random.seed(1337)  # for reproducibility
import random
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.optimizers import RMSprop, SGD, Adadelta, Adagrad
from keras import backend as K
from keras import regularizers       
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils, generic_utils
import h5py
import logging

batch_size = 128
nb_classes = 120

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def to_numerical(categorical):
    """Converts a binary class matrix to a class vector (integers).
    
    # Arguments
        categorical: binary class matrix to be converted into a vector
        
    # Returns
        A vector representation of the input
    """
    numExample = categorical.shape[0]
    y = np.zeros((numExample, 1))
    
    for i in range(numExample):
        y[i, 0] = int(np.where(categorical[i, :] == 1)[0])
    return y


def normalize(x, axis=-1, order=2):
    """Normalizes a Numpy array.

    # Arguments
        x: Numpy array to normalize.
        axis: axis along which to normalize.
        order: Normalization order (e.g. 2 for L2 norm).

    # Returns
        A normalized copy of the array.
    """
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)


def create_pairs(x1, x2, digit_indices_imi, digit_indices_rec, num_classes):
    """Positive and negative pair creation.
    Alternates between positive and negative pairs.
    """
    pairs = []
    labels = []
    pairs_left = []
    pairs_right = []
    
    n = min([len(digit_indices_imi[d]) for d in range(num_classes)])
    #n = (int) (n / 10)
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices_imi[d][i], digit_indices_rec[d][0]
            pairs += [[x1[z1], x2[z2]]]
            pairs_left += [x1[z1]]
            pairs_right += [x2[z2]]
            
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes #classes
            z1, z2 = digit_indices_imi[d][i], digit_indices_rec[dn][0]
            pairs += [[x1[z1], x2[z2]]]
            pairs_left += [x1[z1]]
            pairs_right += [x2[z2]]
            
            labels += [1, 0]
    #return np.array(pairs), np.array(labels)
    return np.array(pairs_left), np.array(pairs_right), np.array(labels)

def create_pairs_pure(x1, x2, digit_indices_imi, digit_indices_rec, num_classes):
    """Positive and negative pair creation.
    Alternates between positive and negative pairs.
    """
    pairs = []
    labels = []
    pairs_left = []
    pairs_right = []
    
    n = min([len(digit_indices_imi[d]) for d in range(num_classes)])
    for d in range(num_classes):
        for i in range(n):
            #print(i)
            z1, z2 = digit_indices_imi[d][i], digit_indices_rec[d][0]
            pairs += [[x1[z1], x2[z2]]]
            pairs_left += [x1[z1]]
            pairs_right += [x2[z2]]
            
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes #classes
            z1, z2 = digit_indices_imi[d][i], digit_indices_rec[dn][0]
            pairs += [[x1[z1], x2[z2]]]
            pairs_left += [x1[z1]]
            pairs_right += [x2[z2]]
            
            labels += [1, 0]
    #return np.array(pairs), np.array(labels)
    return np.array(pairs_left), np.array(pairs_right), np.array(labels)

def create_pairs_retrieval(x1, x2, digit_indices_imi, digit_indices_rec, num_classes):
    """Imitation is paired with every sound concept recording
    From class 1 to 20, each class has 3 imitations in test set, 
    in total 20 * 3 * 20 = 1200 pairs
    """
    pairs = []
    labels = []
    pairs_left = []
    pairs_right = []
    
    #n = min([len(digit_indices_imi[d]) for d in range(num_classes)])
    for d in range(num_classes): # 20 classes
        for i in range(len(digit_indices_imi[d])): # 10 or 11 imitations, len(digit_indices_imi[d] was originally n
            #print(i)
            for j in range(num_classes):
                z1, z2 = digit_indices_imi[d][i], digit_indices_rec[j][0]
                pairs += [[x1[z1], x2[z2]]]
                pairs_left += [x1[z1]]
                pairs_right += [x2[z2]]
                if (d == j):
                    labels += [1]
                else:
                    labels += [0]
            
            #inc = random.randrange(1, classes)
            #dn = (d + inc) % 10
            #z1, z2 = digit_indices_imi[d][i], digit_indices_rec[dn][0]
            #pairs += [[x1[z1], x2[z2]]]
            
            #labels += [1, 0]
    return np.array(pairs_left), np.array(pairs_right), np.array(labels)


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    #return labels[predictions.ravel() < 0.5].mean()
    n = labels.shape[0]
    ii=0
    cout=0
    for ii in range(n):
        if (labels[ii] == (int)(round(predictions[ii]))):
            cout=cout+1
    return cout*1.0/n     


# print(" * Set up the Keras model...")

def get_model():
    global overall_model
    # Convolutional Siamese Network Definition
    # First, define the imitation model
    input_a = Input(shape=(1, 39, 482))
    
    x = Conv2D(48, (6, 6), padding='valid', data_format='channels_first', name='lid_conv_1')(input_a)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(48, (6, 6), padding='valid', data_format='channels_first', name='lid_conv_2')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(48, (6, 6), padding='valid', data_format='channels_first', name='lid_conv_3')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    
    x = Flatten()(x)
    
    output_a = x
    model_a = Model(input_a, output_a)
    # model_a.load_weights('./pretrained_weights/lid_cnn_weights_69_8.h5', by_name=True)

    # Second, define the recording model
    input_b = Input(shape=(1, 128, 128))
    
    y = Conv2D(24, (5, 5), padding='valid', data_format='channels_first', name='nyu_conv_1')(input_b)
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(2, 4))(y)

    y = Conv2D(48, (5, 5), padding='valid', data_format='channels_first', name='nyu_conv_2')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(2, 4))(y)

    y = Conv2D(48, (5, 5), padding='valid', data_format='channels_first', name='nyu_conv_3')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)

    y = Flatten()(y)

    output_b = y
    model_b = Model(input_b, output_b)

    # Then instantiate the imitation and recording model
    voice_input_a = Input(shape=(1, 39, 482))
    voice_input_b = Input(shape=(1, 128, 128))
    voice_output_a = model_a(voice_input_a)
    voice_output_b = model_b(voice_input_b)
    concatenated = keras.layers.concatenate([voice_output_a, voice_output_b])
    out = Dense(108, activation = 'relu')(concatenated)
    out = Dense(1, activation='sigmoid')(out)

    overall_model = Model(inputs = [voice_input_a, voice_input_b], outputs = out)

    # Train the convolutional siamese network
    sgd = SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=True)
    adagrad = Adagrad(lr=0.01)

    # overall_model.summary()
    overall_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model_a.load_weights('./weights/with_pretrain_asym_imi.h5')
    model_b.load_weights('./weights/with_pretrain_asym_rec.h5')
    overall_model.load_weights('./weights/with_pretrain_asym_metric.h5')
    # print(" * Model loaded!")

get_model()

strs = 'ai'
# Prepare the second half data set
dataPredImi = np.load('./log_mel_16k_39_482_data/second_half/'+'/data_'+strs+'.npy')
labelPredImi = np.load('./log_mel_16k_39_482_data/second_half/'+'/label_'+strs+'.npy')
dataPredRec = np.load('./log_mel_44.1k_data/second_half/'+'/dataRec_'+strs+'.npy')
labelPredRec = np.load('./log_mel_44.1k_data/second_half/'+'/labelRec_'+strs+'.npy')

# Create predicting positive and negative pairs
classesPred = 20 # originally labelPredRec.shape[0]

labelPredImi_num = to_numerical(labelPredImi)
labelPredRec_num = to_numerical(labelPredRec)

dataPredImi = dataPredImi.reshape(dataPredImi.shape[0], 39 * 482).astype('float32')
dataPredRec = dataPredRec.reshape(dataPredRec.shape[0], 128 * 128).astype('float32')

# Normalize the data
m4 = np.mean(dataPredImi, axis=1)
m5 = np.mean(dataPredRec, axis=1)
m4 = m4.reshape(m4.shape[0], 1)
m5 = m5.reshape(m5.shape[0], 1)

std4 = np.std(dataPredImi, axis=1)
std5 = np.std(dataPredRec, axis=1)
std4 = std4.reshape(std4.shape[0], 1)
std5 = std5.reshape(std5.shape[0], 1)

dataPredImi = np.multiply(dataPredImi-np.repeat(m4,dataPredImi.shape[1], axis=1), 1./np.repeat(std4, dataPredImi.shape[1], axis=1))
dataPredRec = np.multiply(dataPredRec-np.repeat(m5,dataPredRec.shape[1], axis=1), 1./np.repeat(std5, dataPredRec.shape[1], axis=1))

# Find indices of each class for predicting
digit_indices_pred_imi = [np.where(labelPredImi_num == u)[0] for u in range(classesPred)]
digit_indices_pred_rec = [np.where(labelPredRec_num == v)[0] for v in range(classesPred)]

# Binary classification task for predicting
pred_pairs_left, pred_pairs_right, pred_y = create_pairs_pure(dataPredImi, dataPredRec, 
                                        digit_indices_pred_imi, 
                                        digit_indices_pred_rec, classesPred)

pred_pairs_left = pred_pairs_left.reshape(pred_pairs_left.shape[0], 1, 39, 482)
pred_pairs_right = pred_pairs_right.reshape(pred_pairs_right.shape[0], 1, 128, 128)

# Binary classification task performance on second half dataset
#pred_posi = None
#pred_nega = None

posi_left = pred_pairs_left[0:1,:,:,:]
posi_right = pred_pairs_right[0:1,:,:,:]
gt_1 = pred_y[0:1]
pd_1 = overall_model.predict([posi_left, posi_right])

nega_left = pred_pairs_left[1:2,:,:,:]
nega_right = pred_pairs_right[1:2,:,:,:]
gt_2 = pred_y[1:2]
pd_2 = overall_model.predict([nega_left, nega_right])

a, b, c ,d = float(gt_1[0]), float(pd_1[0][0]), float(gt_2[0]), float(pd_2[0][0])

posi_pair_gt, posi_pair_pd, nega_pair_gt, nega_pair_pd = a, b, c, d

aa = 100
bb = 200
#sys.stdout.flush()
print aa
print bb
print posi_pair_gt
print posi_pair_pd
print nega_pair_gt
print nega_pair_pd
