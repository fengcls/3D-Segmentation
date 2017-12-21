import numpy as np
from keras import backend as K
from keras import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, Conv3DTranspose, Concatenate, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf

global ch_axis
K.set_image_dim_ordering('th') # channels first
if K.image_dim_ordering() is 'th':
    ch_axis = 1
elif K.image_dim_ordering() is 'tf':
    ch_axis = -1

def unet_model_3d(input_shape, first_channel_num=8, depth = 3, pool_size=(2, 2, 2), n_labels=1,
                  initial_learning_rate=0.00001, use_conv_transpose=True, bn_flag = False):
    if K.image_dim_ordering() is 'th':
        inputs = Input(shape=(1, input_shape[0], input_shape[1], input_shape[2]))
    elif K.image_dim_ordering() is 'tf':
        inputs = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1))

    conv1_d = conv3d(first_channel_num, (3, 3, 3), bn_flag)(inputs)
    conv1_d = conv3d(2*first_channel_num, (3, 3, 3), bn_flag)(conv1_d)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1_d)

    conv2_d = conv3d(2*first_channel_num, (3, 3, 3), bn_flag)(pool1)
    conv2_d = conv3d(4*first_channel_num, (3, 3, 3), bn_flag)(conv2_d)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2_d)

    conv3_d = conv3d(4*first_channel_num, (3, 3, 3), bn_flag)(pool2)
    conv3_d = conv3d(8*first_channel_num, (3, 3, 3), bn_flag)(conv3_d)

    #################################################################################################################
    if depth >= 4:          # depth 4, 5
        pool3 = MaxPooling3D(pool_size=pool_size)(conv3_d)

        conv4_d = conv3d(8*first_channel_num, (3, 3, 3), bn_flag)(pool3)
        conv4_d = conv3d(16*first_channel_num, (3, 3, 3), bn_flag)(conv4_d)

        ################################################################################################################
        if depth == 5:      # depth 5
            pool4 = MaxPooling3D(pool_size=pool_size)(conv4_d)

            conv5_d = conv3d(16*first_channel_num, (3, 3, 3), bn_flag)(pool4)
            conv5_d = conv3d(32*first_channel_num, (3, 3, 3), bn_flag)(conv5_d)

            conv5_u = conv5_d
            up4 = upsampling_convtranspose(conv5_u, 32 * first_channel_num, pool_size, use_conv_transpose)
            up4 = Concatenate(axis=ch_axis)([up4, conv4_d])
            conv4_u = conv3d(16*first_channel_num, (3, 3, 3), bn_flag)(up4)
            conv4_u = conv3d(16*first_channel_num, (3, 3, 3), bn_flag)(conv4_u)
        elif depth == 4:    # depth 4
            conv4_u = conv4_d
        ################################################################################################################
        up3 = upsampling_convtranspose(conv4_u, 16 * first_channel_num, pool_size, use_conv_transpose)

        up3 = Concatenate(axis=ch_axis)([up3, conv3_d])
        conv3_u = conv3d(8*first_channel_num, (3, 3, 3), bn_flag)(up3)
        conv3_u = conv3d(8*first_channel_num, (3, 3, 3), bn_flag)(conv3_u)

    #################################################################################################################
    if depth == 3:      # depth 3
        conv3_u = conv3_d

    up2 = upsampling_convtranspose(conv3_u, 8*first_channel_num, pool_size, use_conv_transpose)

    up2 = Concatenate(axis=ch_axis)([up2, conv2_d])
    conv2_u = conv3d(4*first_channel_num, (3, 3, 3), bn_flag)(up2)
    conv2_u = conv3d(4*first_channel_num, (3, 3, 3), bn_flag)(conv2_u)

    up1 = upsampling_convtranspose(conv2_u, 4*first_channel_num, pool_size, use_conv_transpose)

    up1 = Concatenate(axis=ch_axis)([up1, conv1_d])
    conv1_u = conv3d(2*first_channel_num, (3, 3, 3), bn_flag)(up1)
    conv1_u = conv3d(2*first_channel_num, (3, 3, 3), bn_flag)(conv1_u)

    output = Conv3D(n_labels, (1, 1, 1), activation=None)(conv1_u)
    output = Activation('sigmoid')(output)
    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def upsampling_convtranspose(input, filter, pool_size, use_conv_transpose):
    if use_conv_transpose:
        up = Conv3DTranspose(filter, (2, 2, 2), strides = (2,2,2), activation='relu', padding='same')(input)
    else:
        up = UpSampling3D(size=pool_size)(input)
    return up

def conv3d(first_channel_num, kernel_size = (3, 3, 3), bn_flag = False):
    def f(input):
        if bn_flag:
            output = Conv3D(filters=first_channel_num, kernel_size = kernel_size, activation=None, padding='same')(input)
            output = BatchNormalization(axis=ch_axis, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                    beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                    moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                    beta_constraint=None, gamma_constraint=None
                                    )(output)
            output = Activation('relu')(output)

        else:
            output = Conv3D(filters=first_channel_num, kernel_size=(3, 3, 3), activation='relu', padding='same')(input)
        return output
    return f
