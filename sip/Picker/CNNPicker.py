from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout, UpSampling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np


class Model:

    """Class for work with FB prediction model"""


    def __init__(self, ):
        pass


    POOLING = {
        'max': MaxPooling1D,
        'average': AveragePooling1D,
    }


    def conv1d_block(self,
        layer,
        filters,
        kernel_size,
        activation,
        dropout=None,
        batch_norm=True,
        acivation_after_batch=False,
        pooling=None,
        upsampling=None,
        pooling_type='max',
        seed=37
    ):
        x = Conv1D(
            filters,
            kernel_size=kernel_size,
            activation=None,
            padding='same'
        )(layer)
        if acivation_after_batch:
            x = BatchNormalization()(x) if batch_norm else x
            x = Activation(activation)(x)
        else:
            x = Activation(activation)(x)
            x = BatchNormalization()(x) if batch_norm else x

        x = Dropout(dropout, seed=seed)(x) if dropout else x
        x = self.POOLING[pooling_type](pooling) if pooling else x
        x = UpSampling1D(upsampling) if upsampling else x
        return x


    def model_conv1d(
        n_cl,
        shape=None,
        depth=6,
        filters=32,
        kernel_size=32,
        channels=1,
        activation='relu',
        last_activation='softmax',
        last_kernel_size=32,
        last_batch_norm=False,
        last_dropout=None,
        dropout=.1,
        batch_norm=True,
        acivation_after_batch=False,
        pooling=None,
        lr=.001,
        decay=.0,
    ):
        input_img = Input(shape=(shape, channels))

        x = input_img
        for i in range(depth):
            x = Model.conv1d_block(
                x,
                filters,
                kernel_size,
                activation,
                dropout=dropout,
                batch_norm=batch_norm,
               acivation_after_batch=acivation_after_batch,
               pooling=pooling,
               upsampling=None,
               pooling_type='max',
               seed=37,
            )
        x = Model.conv1d_block(
            x,
            n_cl,
            last_kernel_size,
            last_activation,
            dropout=last_dropout,
            batch_norm=last_batch_norm,
            acivation_after_batch=acivation_after_batch,
            pooling=None,
            upsampling=None,
            pooling_type='max',
            seed=37,
        )

        model = Model(input_img, x, name="conv_segm")

        optimizer = Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            amsgrad=False
        )

        loss = 'categorical_crossentropy'
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        return model

## _______________________- check whats above


    def create_model(y_mask, x): ## get y_mask, x from READER/PREPARER
        model = Model.model_conv1d(
           y_mask.shape[-1],
           shape = x.shape[1],
           depth=4,
           filters=32,
           kernel_size=32,
           channels=1,
           activation='relu',
           last_activation='softmax',
           last_kernel_size=32,
           dropout=.5,
           batch_norm=True,
           acivation_after_batch=False,
           pooling=None,
           lr=0.001,
        )
        return model


    def set_checkponits():
        model_checkpoint_callback = ModelCheckpoint(
           filepath='models/weights/new_model_with_random_training_data', ## change path!
           save_weights_only=True,
           monitor='val_accuracy',
           mode='max',
           save_best_only=True
           )
        return model_checkpoint_callback

    
    def fit_model(model, x, y_mask):
        history = model.fit(
        x,
        y_mask,
        epochs=500,
        batch_size=64,
        validation_split=0.375,
        verbose=True,
        callbacks=[Model.set_checkponits()] ## ок сразу вызывать?
        )
        model.load_weights('../fb_picking_notebooks/models/model_weights_whole_area') ## ! change path
        return


    def prediction():
        x, _, y_mask, x_test = SEGYReader.generate_data() #import SEGTReader
        testmodel = Model.create_model(y_mask, x)
        Model.fit_model(testmodel, x, y_mask)
        res = testmodel.predict(x_test)
        lst_pred = [np.argmax(res[i, :, 1]) for i in range(res.shape[0])]
        return lst_pred
