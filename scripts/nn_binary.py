from .nn_resnet import identity_block


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

    y = Dense(2, activation='softmax')(y)

    model = Model(inputs=x, outputs=y)

    model.compile(optimizer=Adam(lr=1e-3), loss=sparse_categorical_crossentropy,
                  metrics=['acc'])
    return model


