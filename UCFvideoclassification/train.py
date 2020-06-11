from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from data import DataSet
from collections import deque
import sys
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import time
import os.path
checkpointer = ModelCheckpoint(
    filepath=os.path.join('../UCF101_video_classification/data', 'checkpoints', 'lstm-features' + '.{epoch:03d}-{val_loss:.3f}.hdf5'),
    verbose=1,
    save_best_only=True)

# Helper: TensorBoard
tb = TensorBoard(log_dir=os.path.join('../UCF101_video_classification/data', 'logs', 'lstm'))

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: Save results.
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('../UCF101_video_classification/data', 'logs', 'lstm' + '-' + 'training-' + \
    str(timestamp) + '.log'))

# Get the data and process it.
data = DataSet(
    seq_length=40,
    class_limit=70
)
#listt=[]
#listt2=[]
X, y = data.get_all_sequences_in_memory('train', 'features')
X_test, y_test = data.get_all_sequences_in_memory('test', 'features')


model = Sequential()
model.add(LSTM(2048, return_sequences=False,input_shape=(40,2048),dropout=0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(data.classes), activation='softmax'))
optimizer = Adam(lr=1e-5, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                    metrics=['accuracy','top_k_categorical_accuracy'])
print(model.summary())

model.fit(
    X,
    y,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[tb, early_stopper, csv_logger,checkpointer],
    epochs=100)
