from tensorflow.keras.datasets import cifar10
import numpy as np
np.random.seed(10)
(x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()
#print(len(x_img_train))#50000
#print(len(x_img_test))#10000
#print(x_img_train.shape) #(50000, 32, 32, 3)
#print(x_img_test[0])
label_dict = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}
import matplotlib.pyplot as plt


def show_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(images[idx], cmap='binary')
        title = str(i) + 'label_dict[labels[i][0]]'
        if len(prediction) > 0:
            title += ',label_dict[prediction[i]]'
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()
#show_images_labels_prediction(x_img_train, y_label_train, [], 0)

##images preprocessing
#print(x_img_train[0][0][0])#[59 62 63]
x_img_train_normalize = x_img_train.astype('float32') / 255
x_img_test_normalize = x_img_test.astype('float32') / 255
#print(x_img_train_normalize[0][0][0])#[0.23137255 0.24313726 0.24705882]
#print(y_label_train.shape)#(50000, 1)
#print(y_label_train[:5])

from tensorflow.keras.utils import to_categorical
from keras import utils as np_utils
y_label_train_Onehot = np_utils.to_categorical(y_label_train)
y_label_test_Onehot = np_utils.to_categorical(y_label_test)
#print(y_label_train_Onehot.shape) #(50000, 10)
#print(y_label_train_Onehot[:5])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Activation, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D

model = Sequential()
model.add(
    Conv2D(filters=32,
           kernel_size=(3, 3),
           input_shape=(32, 32, 3),
           activation='relu',
           padding='same'))
model.add(Dropout(0.25))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.25))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_history = model.fit(x_img_train_normalize,
                          y_label_train_Onehot,
                          validation_split=0.2,
                          epochs=10,
                          batch_size=128,
                          verbose=1)

def show_train_hitory(train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train_History')
    plt.ylabel('train')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_hitory('accuracy', 'val_accuracy')
show_train_hitory('loss', 'val_loss')
#loss: 0.4698 - accuracy: 0.8357 - val_loss: 0.7538 - val_accuracy: 0.7446

scores = model.evaluate(x_img_test_normalize, y_label_test_Onehot, verbose=0)
#print(scores[1])#0.7392
prediction = model.predict_classes(x_img_test_normalize)
#print(prediction[:10])#[3 8 8 0 4 6 1 6 3 1]
show_images_labels_prediction(x_img_test, y_label_test, prediction, 0, 10)

predicted_Probability = model.predict(x_img_test_normalize)
def show_Predicted_Probability(y, prediction, x_img, Predicted_Probability, i):
    print('label:', label_dict[y[i][0]], 'predict', label_dict[prediction[i]])
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(x_img_test[i], (32, 32, 3)))
    plt.show()
    for j in range(10):
        print(label_dict[j] + 'Probability: % 1.9f' %
              (Predicted_Probability[i][j]))

show_Predicted_Probability(y_label_test, prediction, x_img_test, predicted_Probability, 0)

import pandas as pd
pd.crosstab(y_label_test.reshape(-1),
            prediction,
            rownames=['label'],
            colnames=['predict'])













