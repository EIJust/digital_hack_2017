import keras

from keras.models import Sequential


x_train = 0
x_test = 0
y_train = 0
y_test = 0

batch_size = 32
epochs = 2


def get_dataset():
    pass


def dataset_partitioning():
    pass


def dataset_preprocessing():
    pass


if __name__ == '__main__':
    keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None)

    get_dataset()
    dataset_preprocessing()
    dataset_partitioning()

    model = Sequential()

    #Build layers
    pass

    model.compile(loss='binary_croyssentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    print(u'Train model...')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size, verbose=1)
    print(u'Score: {}'.format(score[0]))
    print(u'Accuracy: {}'.format(score[1]))
