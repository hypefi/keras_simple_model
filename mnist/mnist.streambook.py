
import streamlit as __st
import streambook
__toc = streambook.TOCSidebar()


__toc.generate()
with __st.echo(), streambook.st_stdout('info'):
    #Plot ad hos mnist instances
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.utils import np_utils
with __st.echo(), streambook.st_stdout('info'):
    import matplotlib.pyplot as plt
with __st.echo(), streambook.st_stdout('info'):
    #load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
with __st.echo(), streambook.st_stdout('info'):
    #flatter image
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
__st.markdown(r"""normalize inputs """, unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    X_train = X_train / 255
    X_test = X_test / 255
__st.markdown(r"""one hot encode outputs""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    y_train = np_utils.to_categorical(y_train)
    y_test= np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
__st.markdown(r"""define baseline model""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    def baseline_model():
        #create model
        model = Sequential()
        model.add(Dense(num_pixels, input_dim= num_pixels, kernel_initializer='normal', activation='relu'))
        model.add(Dense(num_classes, kernel_initializer='normal', activation= 'softmax'))
        #Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
__st.markdown(r"""build the model""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    model = baseline_model()
__st.markdown(r"""fit the model""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
__st.markdown(r"""Final evaluation of the model""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    scores = model.evaluate(X_test, y_test, verbose=0)
with __st.echo(), streambook.st_stdout('info'):
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
with __st.echo(), streambook.st_stdout('info'):
    """
    #plot 4 images as gray scale
    plt.subplot(221)
    plt.imshow(X_train[0], cmap = plt.get_cmap('gray'))

    plt.subplot(222)
    plt.imshow(X_train[1], cmap = plt.get_cmap('gray'))

    plt.subplot(223)
    plt.imshow(X_train[2], cmap = plt.get_cmap('gray'))

    plt.subplot(224)
    plt.imshow(X_train[3], cmap = plt.get_cmap('gray'))

    #show the plot

    plt.show()
    """

