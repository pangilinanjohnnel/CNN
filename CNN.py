import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, VGG16
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

IMG_SIZE = (96, 96)
BATCH_SIZE = 8
EPOCHS = 3

base_path = r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\advance data science\to do\mri"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_path + r"\train", validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_path + r"\train", validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_path + r"\test", image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
for i, name in enumerate(class_names):
    print(f"Index {i}: {name}")

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).cache().shuffle(100).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(tf.data.AUTOTUNE)

#EVALUATION
def evaluate_model(model, name):
    test_ds_local = test_ds.unbatch().batch(BATCH_SIZE)
    y_true = np.concatenate([y for x, y in test_ds_local], axis=0)
    y_probs = model.predict(test_ds_local)
    y_pred = np.argmax(y_probs, axis=1)

    print(f"\n{'='*20}\n{name} Report\n{'='*20}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))
    auc = roc_auc_score(y_bin, y_probs, multi_class='ovr', average='macro')
    print(f"{name} ROC-AUC: {auc:.4f}")

#CNN
cnn_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(96,96,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("CNN")
cnn_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

base_mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96,96,3))
base_mobilenet.trainable = False  # Freeze weights

model_fe = models.Sequential([
    base_mobilenet,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model_fe.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nTraining MobileNetV2 (Feature Extraction)...")
model_fe.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

#MOBILENET V2
model_ft = models.clone_model(model_fe)
model_ft.set_weights(model_fe.get_weights())

model_ft.layers[0].trainable = True
for layer in model_ft.layers[0].layers[:-20]:
    layer.trainable = False

model_ft.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("MobileNetV2 (Fine-Tuning)")
model_ft.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

#VGG1
base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(96,96,3))
base_vgg.trainable = False

model_vgg = models.Sequential([
    base_vgg,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model_vgg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training VGG16 (Pre-trained Model 2)...")
model_vgg.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# --- FINAL COMPARISON ---
evaluate_model(cnn_model, "Scratch CNN")
evaluate_model(model_fe, "MobileNetV2 (Feature Extraction)")
evaluate_model(model_ft, "MobileNetV2 (Fine-Tuning)")
evaluate_model(model_vgg, "VGG16 (Pre-trained 2)")