from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import *
from keras import regularizers
from .assessment import calc_f1


def identity_block(input_tensor, filters, reduce_dim=False):
    l2 = 1e-4
    f1, f2, f3 = filters
    x = Conv1D(f1, 1, padding='same', use_bias=0, kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(l2))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if not reduce_dim:
        x = Conv1D(f2, 3, padding='same', use_bias=0, kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(l2))(x)
    else:
        x = Conv1D(f2, 3, padding='same', strides=2, use_bias=0, kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(l2))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(f3, 1, padding='same', use_bias=0, kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(l2))(x)
    x = BatchNormalization()(x)

    if reduce_dim:
        input_tensor = Conv1D(f3, 1, padding='same', kernel_initializer='he_normal',
                              strides=2, use_bias=0,
                              kernel_regularizer=regularizers.l2(l2))(input_tensor)
        input_tensor = BatchNormalization()(input_tensor)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def resnet(num_class, input_shape=(2600, 1)):
    x = Input(shape=input_shape)
    y = x

    y = ZeroPadding1D(3)(y)
    y = Conv1D(16, 7, padding='valid', use_bias=0, strides=2,
               kernel_initializer='he_normal')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = ZeroPadding1D(1)(y)
    y = MaxPool1D(3, padding='valid', strides=2)(y)

    y = identity_block(y, [16, 16, 64], 1)
    y = identity_block(y, [16, 16, 64])
    y = identity_block(y, [16, 16, 64])

    y = identity_block(y, [32, 32, 128], 1)
    y = identity_block(y, [32, 32, 128])
    y = identity_block(y, [32, 32, 128])

    y = identity_block(y, [64, 64, 256], 1)
    y = identity_block(y, [64, 64, 256])
    y = identity_block(y, [64, 64, 256])

    # y = identity_block(y, [128, 128, 512], 1)
    # y = identity_block(y, [128, 128, 512])
    # y = identity_block(y, [128, 128, 512])

    y = GlobalAveragePooling1D()(y)

    y = Dense(num_class, activation='softmax')(y)

    model = Model(inputs=x, outputs=y)

    model.compile(optimizer=Adam(lr=1e-3), loss=sparse_categorical_crossentropy,
                  metrics=['acc'])
    return model
