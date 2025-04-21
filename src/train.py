import time
import tensorflow as tf


def main():
    BATCH_SIZE = 32
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    SEED = int(time.time())
    VALIDATION_SPLIT = 0.1
    EPOCHS = 20
    DS_DIR = "dataset/"

    # load dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DS_DIR,
        seed=SEED,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DS_DIR,
        seed=SEED,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Class names = {class_names}")

    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # create model
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0/255, input_shape=(28, 28, 3)),
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            input_shape=(28, 28, 3),
            activation='relu',
            padding='same'
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            activation='relu',
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(num_classes)
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    # train model
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    model.save_weights(f"saves/model-{int(time.time())}.weights.h5")


if __name__ == "__main__":
    main()
