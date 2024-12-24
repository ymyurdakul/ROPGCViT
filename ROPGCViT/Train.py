import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, roc_auc_score

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU!")

from keras_cv_attention_models import gcvit

# Parameters
batch_size = 64
img_width = 224
img_height = 224
learning_rate = 1e-3
momentum = 0.9
weight_decay = 1e-4
epochs = 50
base_save_dir = "Results"


model = gcvit.ROPGCViT(pretrained="imagenet", num_classes=5)
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, decay=weight_decay),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


if not os.path.exists(base_save_dir):
    os.makedirs(base_save_dir)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Folds (Only one)
for fold_num in range(1, 2):
    data_path = f'foldlar/fold_{fold_num}'
    path_train = f"{data_path}/train"
    path_test = f"{data_path}/test"
    path_val = f"{data_path}/val"

    train_generator = train_datagen.flow_from_directory(
        directory=path_train,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical"
    )

    valid_generator = test_datagen.flow_from_directory(
        directory=path_val,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical"
    )

    test_generator = test_datagen.flow_from_directory(
        directory=path_test,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    # Model saving
    save_dir = os.path.join(base_save_dir, f"fold_{fold_num}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_save_path = os.path.join(save_dir, "model_fold.h5")
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True)

    # Train
    start_train_time = time.time()
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=valid_generator,
        callbacks=[checkpoint, early_stopping]
    )
    train_duration = time.time() - start_train_time

    # Test
    start_test_time = time.time()
    y_pred_prob = model.predict(test_generator)
    test_duration = time.time() - start_test_time

    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true_ohe = tf.keras.utils.to_categorical(test_generator.classes, num_classes=5)

    # Save Results
    metrics_save_path = os.path.join(save_dir, "classification_metrics.txt")
    with open(metrics_save_path, 'w') as f:
        kappa = cohen_kappa_score(test_generator.classes, y_pred)
        auc_val = roc_auc_score(y_true_ohe, y_pred_prob, multi_class='ovr')
        f.write(f"Cohen's Kappa: {kappa}\n")
        f.write(f"AUC: {auc_val}\n")
        f.write(f"Training Time: {train_duration} seconds\n")
        f.write(f"Testing Time: {test_duration} seconds\n")

        cm = confusion_matrix(test_generator.classes, y_pred)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

        report = classification_report(test_generator.classes, y_pred, digits=7)
        f.write("\n\nClassification Report:\n")
        f.write(report)

    # Graphics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
    plt.close()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()

    df_cm = pd.DataFrame(cm, index=np.arange(cm.shape[0]), columns=np.arange(cm.shape[1]))
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
