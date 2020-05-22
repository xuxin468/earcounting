from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from  tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import  Dropout, Flatten, Dense, LocallyConnected2D, BatchNormalization
from tensorflow.keras import backend as K
import pickle

def ear_model(img_width,img_height,log_dir,epochs,train_generator,validation_generator,
              nb_train_samples, nb_validation_samples,batch_size):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=input_shape,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (3, 3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

    model.add(Conv2D(256, (3, 3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

    model.add(Conv2D(512, (3, 3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

    model.add(Conv2D(512, (3, 3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(4,activation='softmax'))


    model.compile(
        loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    logging = TensorBoard(log_dir=log_dir, write_images=True)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-accuracy{acc:.4f}-val_acc{val_acc:.4f}.h5', monitor='val_acc',
                                 # save_weights_only=True,
                                 save_best_only=True, period=5)
    reduce_lr = ReduceLROnPlateau(monitor='lr', factor=0.1, patience=8, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1)
    H=model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        initial_epoch=0,
        validation_data=validation_generator,
        callbacks=[logging,checkpoint,reduce_lr,early_stopping],
        validation_steps=nb_validation_samples // batch_size)
    model.save(log_dir + 'ear.h5')
    return  model
