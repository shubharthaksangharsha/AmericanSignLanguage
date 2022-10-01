import tensorflow as tf

train_dir = "./Data/train"
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.,
                                                                validation_split=0.1)

train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               target_size=(224, 224),
                                               class_mode="categorical",
                                               subset="training",
                                               seed=42)

test_data = train_datagen.flow_from_directory(directory=train_dir,
                                               target_size=(224, 224),
                                               class_mode="categorical",
                                               subset="validation",
                                               seed=42)

tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, 3, input_shape=(224, 224, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPool2D(3, 3, padding="same"),
    tf.keras.layers.Conv2D(32, 3, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPool2D(3, 3, padding="same"),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=4, activation="softmax"),
])



