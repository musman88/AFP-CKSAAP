from keras.models import Sequential
from keras.utils import np_utils
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

def RAFP_model(input_dim, nb_classes):
    inputs = Input([input_dim,1])
    layer0 = Flatten()(inputs)

    layer1 = Dense(128, activation='relu', name="layer1")(layer0)
    layer1 = Dropout(0.7)(layer1)

    layer2 = Dense(128, activation='relu', name="layer2")(layer1)
    layer2 = Dropout(0.7)(layer2)

    layer3 = Dense(128, activation='relu', name="layer3")(layer2)
    layer3 = Dropout(0.7)(layer3)

    layer4 = Dense(128, activation='relu', name="layer4")(layer3)
    layer4 = Dropout(0.7)(layer4)

    # layer5 = Dense(128, activation='relu', name="layer5")(layer4)
    # layer5 = Dropout(0.3)(layer5)

    # layer6 = Dense(128, activation='relu', name="layer6")(layer5)
    # layer6 = Dropout(0.3)(layer6)

    # layer7 = Dense(128, activation='relu', name="layer7")(layer6)
    # layer7 = Dropout(0.3)(layer7)

    # layer8 = Dense(128, activation='relu', name="layer8")(layer7)
    # layer8 = Dropout(0.3)(layer8)

    layer9 = Dense(nb_classes, activation='softmax', name="layer9")(layer4)

    model = Model(input=inputs, output=layer9)
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def RAFP_model_Skip(input_dim, nb_classes):
    inputs = Input([input_dim,1])
    layer0 = Flatten()(inputs)
    layer1 = Dense(70, activation='relu')(layer0)
    layer1 = Dropout(0.7)(layer1)

    # layer2 = concatenate([layer0, layer1], axis=1)
    layer2 = Dense(128, activation='relu')(layer1)
    layer2 = Dropout(0.7)(layer2)

    # layer3 = concatenate([layer0,layer1,layer2], axis=1)
    # layer3 = Dense(128, activation='relu')(layer3)
    # layer3 = Dropout(0.7)(layer3)

    # layer4 = concatenate([layer0,layer1,layer2, layer3], axis=1)
    # layer4 = Dense(128, activation='relu')(layer4)
    # layer4 = Dropout(0.7)(layer4)

    # layer5 = concatenate([layer0, layer1, layer2 , layer3 , layer4], axis=1)
    # layer5 = Dense(128, activation='relu')(layer5)
    # layer5 = Dropout(0.7)(layer5)

    # layer5 = concatenate([layer0, layer1], axis=1) ####
    # layer5 = concatenate([layer1, layer3], axis=1)####
    # layer5 = concatenate([layer2, layer4], axis=1)####
    # layer5 = concatenate([layer0, layer4], axis=1)
    # layer5 = concatenate([layer0, layer5], axis=1)
    # layer5 = concatenate([layer1, layer5], axis=1)
    # layer5 = concatenate([layer2, layer5], axis=1)
    # layer5 = concatenate([layer3, layer5], axis=1)
    # layer6 = concatenate([layer0, layer1, layer2, layer3, layer4, layer5])
    layer6 = concatenate([layer0, layer1, layer2])
    layer6 = Dense(nb_classes, activation='softmax', name="layer6")(layer2)

    model = Model(input=inputs, output=layer6)

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    return model
