from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocess import load_data, augment_data
from model import create_model

def train_model(data_dir="data/train", val_dir="data/val"):
    train_images, train_labels = load_data(data_dir)
    val_images, val_labels = load_data(val_dir)

    model = create_model(input_shape=(48, 48, 1), num_classes=len(set(train_labels)))

    early_stopping = EarlyStopping(monitor="val_loss", patience=5)
    checkpoint = ModelCheckpoint("emotion_model.h5", save_best_only=True, monitor="val_loss")

    train_gen = augment_data(train_images, train_labels)

    model.fit(
        train_gen,
        validation_data=(val_images, val_labels),
        epochs=20,
        callbacks=[early_stopping, checkpoint]
    )

if __name__ == "__main__":
    train_model()
