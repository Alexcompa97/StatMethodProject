import tensorflow as tf
import matplotlib.pyplot as mpplt
import keras_tuner as kt
import pandas
import os
import numpy
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold

os.chdir('e:\ScuolaUni\MagistraleUNIMI\StatMethod\Progetto\DataSet')

batch_size = 16
control = 1000
first_time = True

#5-fold cross validation parameters
      
VALIDATION_ACCURACY = []
VALIDATION_LOSS = []

#Setting parameters of EarlyStopping callback to stop training when loss stops improving

callback = [EarlyStopping(monitor = "val_loss", patience = 5)]

#Creating lists with images from all the sets, then separating them in chihuahua and muffin based on their folder name

img_list = [
    os.path.join(dirname, filename)
    for dirname, _, filenames in os.walk(".")
    for filename in filenames
    if ".jpg" in filename
]
     
chihuahua_imgs = [img for img in img_list if "chihuahua" in img]
muffin_imgs = [img for img in img_list if "muffin" in img]

print("Number of chihuahua images: ", len(chihuahua_imgs))
print("Number of muffin images: ", len(muffin_imgs), '\n')

#Creating data frame using previously created image list

img_labels = [img.split("\\")[2] for img in img_list]
img_paths = [img for img in img_list]

data_frame = pandas.DataFrame({"file path": img_paths, "label": img_labels})

print(data_frame, '\n')

#Creating training, test and validation sets using the previously created data frame

train_valid_set, test_set = train_test_split(data_frame, test_size = 0.20, random_state = 100)
train_set, valid_set = train_test_split(train_valid_set, test_size = 0.15, random_state = 100)

print("Training set:", len(train_set))
print("Validation set:", len(valid_set))
print("Test set:", len(test_set), '\n')


def use_simple_model(batch_size, train_set, valid_set, test_set, callback):

    #Processing images in batches to RGB scale and set size

    datagen = ImageDataGenerator(rescale = 1./255)

    train_gen = datagen.flow_from_dataframe(dataframe = train_set, x_col = "file path", y_col = "label", target_size = (224, 224), batch_size = batch_size, class_mode = "binary", color_mode= "rgb", shuffle = True)
    valid_gen = datagen.flow_from_dataframe(dataframe = valid_set, x_col = "file path", y_col = "label", target_size = (224, 224), batch_size = batch_size, class_mode = "binary", color_mode= "rgb", shuffle = True)

    #Visualizing processed images with corresponding one-hot encoding (1.0 for muffins and 0.0 for chihuahuas)

    images, labels = train_gen.next()

    for i in range(batch_size):
        mpplt.subplot(4, 4, i + 1)
        mpplt.imshow(images[i])
        mpplt.title(labels[i])
        mpplt.axis('off')

    mpplt.tight_layout()
    mpplt.show()

    #Building the simple model

    simple_model = Sequential([

        Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (224, 224, 3)),
        MaxPooling2D(pool_size = (2, 2), strides = 2),

        Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
        MaxPooling2D(pool_size = (2, 2), strides = 2),

        Flatten(),
        Dense(units = 64, activation = 'relu'),

        Dense(units = 1, activation = 'sigmoid'),
    ])

    simple_model.summary()

    simple_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #Train model and show history showing results of training 

    history_simple = simple_model.fit(x = train_gen, validation_data = valid_gen, epochs = 50, callbacks = callback)

    recap_df_simple = pandas.DataFrame(history_simple.history)

    print('\n\n')
    print(recap_df_simple)

    recap_df_simple[["loss","val_loss"]].plot()
    recap_df_simple[["accuracy","val_accuracy"]].plot()

    mpplt.tight_layout()
    mpplt.show()

    predict(test_set, datagen, simple_model)


def data_aug_prep(batch_size, train_set, valid_set, callback, control, first_time, test_set):

    if first_time == True:

        first_time = False

        #Processing images and using data augmentation on them to add more data to training set

        datagen_aug = ImageDataGenerator(rescale = 1./255, rotation_range = 10, width_shift_range = 0.3, height_shift_range = 0.3, shear_range = 0.15, zoom_range = 0.2, channel_shift_range = 10., horizontal_flip = True)

        train_gen_aug = datagen_aug.flow_from_dataframe(dataframe = train_set, x_col = "file path", y_col = "label", target_size = (224, 224), batch_size = batch_size, class_mode = "binary", color_mode ='rgb', shuffle = True)
        valid_gen_aug = datagen_aug.flow_from_dataframe(dataframe = valid_set, x_col = "file path", y_col = "label", target_size = (224, 224), batch_size = batch_size, class_mode = "binary", color_mode ='rgb', shuffle = True)

        #Visualizing processed images with corresponding one-hot encoding (1.0 for muffins and 0.0 for chihuahuas)

        images, labels = train_gen_aug.next()

        for i in range(batch_size):
            mpplt.subplot(4, 4, i + 1)
            mpplt.imshow(images[i])
            mpplt.title(labels[i])
            mpplt.axis('off')

        mpplt.tight_layout()
        mpplt.show()

    match control:
        case '2':
            use_data_aug_model(callback, train_gen_aug, valid_gen_aug, datagen_aug, test_set)
        case '3':
            use_dropout_model(callback, train_gen_aug, valid_gen_aug, datagen_aug, test_set)
        case '4':
            use_hp_tuning(callback, train_gen_aug, valid_gen_aug, datagen_aug, test_set)



def use_data_aug_model(callback, train_gen_aug, valid_gen_aug, datagen_aug, test_set):

    #Building the data augmentation model

    model_data_aug = Sequential([

        Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (224, 224, 3)),
        MaxPooling2D(pool_size = (2, 2), strides = 2),

        Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
        MaxPooling2D(pool_size = (2, 2), strides = 2),

        Flatten(),
        Dense(units = 64, activation = 'relu'),

        Dense(units = 1, activation = 'sigmoid'),
    ])

    model_data_aug.summary()

    model_data_aug.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #Train model and show history showing results of training 

    history_aug = model_data_aug.fit(x = train_gen_aug, validation_data = valid_gen_aug, epochs = 50, callbacks = callback)

    recap_df_aug = pandas.DataFrame(history_aug.history)

    print('\n\n')
    print(recap_df_aug) 

    recap_df_aug[["loss","val_loss"]].plot()
    recap_df_aug[["accuracy","val_accuracy"]].plot()

    mpplt.tight_layout()
    mpplt.show()

    predict(test_set, datagen_aug, model_data_aug)


def use_dropout_model(callback, train_gen_aug, valid_gen_aug, datagen_aug, test_set):

    #Building the dropout layer model

    model_dropout = Sequential([

        Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (224, 224, 3)),
        MaxPooling2D(pool_size = (2, 2), strides = 2),
        Dropout(0.1),

        Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
        MaxPooling2D(pool_size = (2, 2), strides = 2),
        Dropout(0.3),

        Flatten(),
        Dense(units = 64, activation = 'relu'),
        Dropout(0.3),

        Dense(units = 1, activation = 'sigmoid'),
    ])

    model_dropout.summary()

    model_dropout.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #Train model and show history showing results of training 

    history_dropout = model_dropout.fit(x = train_gen_aug, validation_data = valid_gen_aug, epochs = 50, callbacks = callback)

    recap_df_dropout = pandas.DataFrame(history_dropout.history)

    print('\n\n')
    print(recap_df_dropout) 

    recap_df_dropout[["loss","val_loss"]].plot()
    recap_df_dropout[["accuracy","val_accuracy"]].plot()

    mpplt.tight_layout()
    mpplt.show()

    predict(test_set, datagen_aug, model_dropout)


def model_builder(hp):

    #Build model based on dropout layer model

    input_dropout = hp.Float("dropout_input_layer", min_value = 0.05, max_value = 0.2, step = 0.05)
    hidden_dropout = hp.Float("dropout_hidden_layer", min_value = 0.2, max_value = 0.5, step = 0.1)

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters = hp.Int("conv1_filters", min_value = 16, max_value = 64, step = 16), kernel_size = (3, 3), input_shape = (224, 224, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(input_dropout))

    model.add(keras.layers.Conv2D(filters = hp.Int("conv2_filters", min_value = 32, max_value = 128, step = 32), kernel_size = (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(hidden_dropout))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units = hp.Int("num_units", min_value = 64, max_value = 256, step = 64), activation = "relu"))
    model.add(Dropout(hidden_dropout))

    model.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model


def use_hp_tuning(callback, train_gen_aug, valid_gen_aug, datagen_aug, test_set):
        
    tuner = kt.BayesianOptimization(model_builder, objective = 'val_accuracy', max_trials = 4, directory = 'e:\ScuolaUni\MagistraleUNIMI\StatMethod\Progetto', project_name = 'HPtuning', overwrite = False)

    print('\n')
    tuner.search_space_summary()

    #Do not run if tuned hyperparameters already saved
    tuner.search(train_gen_aug, validation_data = valid_gen_aug, epochs = 50, callbacks = callback)

    #Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

    print('\n\n')
    print(best_hps.values)

    hypermodel = tuner.hypermodel.build(best_hps)

    history_hp = hypermodel.fit(x = train_gen_aug, validation_data = valid_gen_aug, epochs = 50, callbacks = callback)
    recap_df_hp = pandas.DataFrame(history_hp.history)

    print('\n\n')
    print(recap_df_hp)

    recap_df_hp[["loss","val_loss"]].plot()
    recap_df_hp[["accuracy","val_accuracy"]].plot()

    mpplt.tight_layout()
    mpplt.show()

    predict(test_set, datagen_aug, hypermodel)

    use_cross_valid(datagen_aug, tuner)


def predict (test_set, datagen, model):

    #Makes prediction based on test set and then creates confusion matrix to visualize model accuracy

    test_gen = datagen.flow_from_dataframe(dataframe = test_set, x_col = "file path", y_col = "label", target_size = (224, 224), class_mode = "binary", color_mode= "rgb", shuffle = False)

    predictions = model.predict(test_gen)

    print('\n\n')
    print(predictions)

    rounded_predictions = [0 if val < 0.5 else 1 for val in predictions]

    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    report = metrics.classification_report(true_classes, rounded_predictions, target_names = class_labels)
    print('\n\n')
    print(report)

    cm = confusion_matrix(true_classes, rounded_predictions)

    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)

    cm_display.plot(include_values = True, ax = None, xticks_rotation = 'horizontal')
    mpplt.show()


def use_cross_valid(datagen_aug, tuner):

    k_fold = KFold(n_splits = 5, random_state = 100, shuffle = True)

    fold_var = 1

    for train_index, val_index in k_fold.split(data_frame):
        training = data_frame["file path"][train_index]
        validation = data_frame["file path"][val_index]

        train_data_frame = data_frame.loc[data_frame["file path"].isin(training)]
        valid_data_frame = data_frame.loc[data_frame["file path"].isin(validation)]

        train_data_gen = datagen_aug.flow_from_dataframe(train_data_frame, x_col = "file path", y_col = "label", class_mode = "binary", shuffle = True, target_size = (224, 224))
        valid_data_gen = datagen_aug.flow_from_dataframe(valid_data_frame, x_col = "file path", y_col = "label", class_mode = "binary", shuffle = True, target_size = (224, 224))

        best_hyperparameters = tuner.get_best_hyperparameters(num_trials = 1)[0]
        model = tuner.hypermodel.build(best_hyperparameters)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), loss = "binary_crossentropy", metrics = ["accuracy"])

        model.fit(train_data_gen, epochs = 50, callbacks = callback, validation_data = valid_data_gen)

        model_path = "e:\ScuolaUni\MagistraleUNIMI\StatMethod\Progetto\Models\model" + str(fold_var) + ".h5"
        model.save(model_path)

        #Load best model to evaluate its performance
        if os.path.exists(model_path):
            model.load_weights(model_path)
        else:
            print("File doesn't exist")

        results = model.evaluate(valid_data_gen)
        results = dict(zip(model.metrics_names, results))

        VALIDATION_ACCURACY.append(results['accuracy'])
        VALIDATION_LOSS.append(results['loss'])

        tf.keras.backend.clear_session()

        fold_var += 1
    
    #Visualize cross validation results

    print('\n\n')
    print(VALIDATION_ACCURACY)
    print(VALIDATION_LOSS)

    print('\n\n')
    print(numpy.mean(VALIDATION_ACCURACY))
    print(numpy.mean(VALIDATION_LOSS))

    mpplt.figure(figsize = (10, 6))

    mpplt.subplot(2, 1, 1)
    mpplt.plot(range(1, len(VALIDATION_ACCURACY) + 1), VALIDATION_ACCURACY, marker = 'o', linestyle = '-', color = 'b')
    mpplt.title('Validation Accuracy per Fold')
    mpplt.xlabel('Fold')
    mpplt.ylabel('Accuracy')
    mpplt.grid(True)

    mpplt.subplot(2, 1, 2)
    mpplt.plot(range(1, len(VALIDATION_LOSS) + 1), VALIDATION_LOSS, marker = 'o', linestyle = '-', color = 'r')
    mpplt.title('Validation Loss per Fold')
    mpplt.xlabel('Fold')
    mpplt.ylabel('Loss')
    mpplt.grid(True)

    mpplt.tight_layout()
    mpplt.show()

#Control to choose model to train

while True:
    control = input ("insert 1 for simple model, 2 for data augmentation model, 3 for dropout layer model or 4 for hyperparameter tuning, insert 0 to quit:\n")

    match control:
        case '0':
            break
        case '1':
            use_simple_model(batch_size, train_set, valid_set, test_set, callback)
        case '2':
            data_aug_prep(batch_size, train_set, valid_set, callback, control, first_time, test_set)
        case '3':
            data_aug_prep(batch_size, train_set, valid_set, callback, control, first_time, test_set)
        case '4':
            data_aug_prep(batch_size, train_set, valid_set, callback, control, first_time, test_set)
        case _:    
            continue           
