import tensorflow as tf
from tensorflow.keras import layers


class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 cnn_filters=500,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.5,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        # self.embedding = layers.Embedding(vocabulary_size,
        #                                   embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu",
                                        input_shape=(510, 768,))
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=6,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=12,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.last_dense = layers.Dense(units=model_output_classes,
                                       activation="softmax")
        '''
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")
        '''

    def call(self, inputs, training):
        l = inputs
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output
