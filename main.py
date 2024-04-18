import os
import glob
import shutil
import cv2
import json
import keras
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0
from keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, BatchNormalization
import warnings
warnings.simplefilter("ignore")

# Defining the working directories

# work_dir = '../input/cassava-leaf-disease-classification/'
# os.listdir(work_dir) 

train_path = './Dataset/train_images'
WORK_DIR = './Dataset'
os.listdir(WORK_DIR)
train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_labels.head()

with open(os.path.join(WORK_DIR, "label_num_to_disease_map.json")) as file:
    print(json.dumps(json.loads(file.read()), indent=4))
print('Train images: %d' %len(os.listdir(
    os.path.join(WORK_DIR, "train_images"))))
print('Cleaned images: %d' %len(os.listdir(
    os.path.join(WORK_DIR, "new_train"))))
label_data = pd.read_csv("./Dataset/train.csv")

with open("./Dataset/label_num_to_disease_map.json", "r") as json_file:
    label_mapping = json.load(json_file)
label_names = [label_mapping[str(label)] for label in sorted(label_mapping.keys(), key=int)]

# Create a new figure
plt.figure(figsize=(10, 6))

# Create a count plot with numerical x-axis and a legend mapping numbers to disease names
ax = sns.countplot(data=label_data, x='label', order=sorted(label_data['label'].unique()), palette='viridis')
ax.set_xticklabels(sorted(label_mapping.keys()), rotation=45, ha='right')
ax.set_xlabel('Classes')
ax.set_ylabel('Count')
ax.set_title('Count of Each Label')

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, xytext=(0, 5),
                textcoords='offset points')

# Create legend with the disease names
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, label_mapping.values(), title="Disease")

plt.tight_layout()
plt.legend(label_names, title='Disease')
plt.show()
# Some photos of "0": "Cassava Bacterial Blight (CBB)"


sample = train_labels[train_labels.label == 0].sample(3)
plt.figure(figsize=(15, 5))
for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(1, 3, ind + 1)
    img = cv2.imread(os.path.join(WORK_DIR, "train_images", image_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    
plt.show()

#Some photos of "1": "Cassava Brown Streak Disease (CBSD)"
sample = train_labels[train_labels.label == 1].sample(3)
plt.figure(figsize=(15, 5))
for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(1, 3, ind + 1)
    img = cv2.imread(os.path.join(WORK_DIR, "train_images", image_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    
plt.show()


#Some photos of "2": "Cassava Green Mottle (CGM)"
sample = train_labels[train_labels.label == 2].sample(3)
plt.figure(figsize=(15, 5))
for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(1, 3, ind + 1)
    img = cv2.imread(os.path.join(WORK_DIR, "train_images", image_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    
plt.show()

#Some photos of "3": "Cassava Mosaic Disease (CMD)"
sample = train_labels[train_labels.label == 3].sample(3)
plt.figure(figsize=(15, 5))
for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(1, 3, ind + 1)
    img = cv2.imread(os.path.join(WORK_DIR, "train_images", image_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    
plt.show()

#Some photos of "4": "Healthy"¶
sample = train_labels[train_labels.label == 4].sample(3)
plt.figure(figsize=(15, 5))
for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(1, 3, ind + 1)
    img = cv2.imread(os.path.join(WORK_DIR, "train_images", image_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    
plt.show()
# baseline model for this task

baseline_pred = [3] * len(label_data)

baseline_accuracy = (label_data['label'] == baseline_pred).mean()
print(baseline_accuracy)

train_labels.label = train_labels.label.astype('str')
BATCH_SIZE = 32
STEPS_PER_EPOCH = len(train_labels)*0.8 / BATCH_SIZE
VALIDATION_STEPS = len(train_labels)*0.2 / BATCH_SIZE
EPOCHS = 20
TARGET_SIZE = 224

train_generator = ImageDataGenerator(
    validation_split= 0.2,
    rescale= 1./255,
    rotation_range= 30,
    width_shift_range= 0.2,
    height_shift_range= 0.2,
    brightness_range= [0.8, 1.2],
    zoom_range= 0.2,
    shear_range= 0.2,
    channel_shift_range= 50,
    horizontal_flip= True,
    vertical_flip= True,
    fill_mode= 'nearest'
    
)

train_generator = train_generator.flow_from_dataframe(
    dataframe=train_labels,
    directory=os.path.join(WORK_DIR, "train_images"),
    subset="training",
    x_col="image_id",
    y_col="label",
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse"
)

validation_generator = ImageDataGenerator(validation_split = 0.2) \
    .flow_from_dataframe(train_labels,
                         directory = os.path.join(WORK_DIR, "train_images"),
                         subset = "validation",
                         x_col = "image_id",
                         y_col = "label",
                         target_size = (TARGET_SIZE, TARGET_SIZE),
                         batch_size = BATCH_SIZE,
                         class_mode = "sparse")



def create_model():
    model = models.Sequential()

    # Adding EfficientNetB0 as the base model
    model.add(EfficientNetB0(
        include_top=False,  # Do not include the top (classification) layer
        weights='imagenet',  # Use weights pre-trained on ImageNet
        input_shape=(TARGET_SIZE, TARGET_SIZE, 3)  # Define the input shape
    ))

    # Adding a global average pooling layer to reduce the spatial dimensions
    model.add(layers.GlobalAveragePooling2D())

    # Adding additional layers for more complex learning patterns
    model.add(layers.Dense(256, activation="relu"))  # First Dense layer with ReLU activation
    model.add(BatchNormalization())  # Normalize activations of the previous layer
    model.add(Dropout(0.5))  # Add dropout to prevent overfitting

    model.add(layers.Dense(128, activation="relu"))  # Another Dense layer with ReLU activation
    model.add(BatchNormalization())  # Normalize activations of the previous layer
    model.add(Dropout(0.3))  # Add dropout to prevent overfitting

    # Adding a fully connected output layer with 5 outputs for your 5 classes
    model.add(layers.Dense(5, activation="softmax"))

    # Compiling the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Set learning rate
        loss="sparse_categorical_crossentropy",  # Suitable loss for integer labels
        metrics=["accuracy"]  # Use 'accuracy' metric
    )
    
    return model


model = create_model()
model.summary()


# Callbacks for early stopping to halt the training early if the validation loss stops improving
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001,
                           patience=4, mode='min', verbose=1,
                           restore_best_weights=True)

# Callback for reducing the learning rate when the validation loss plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_delta=0.001,
                              mode='min', verbose=1)

# Callback to save the best model based on the validation loss
model_save = ModelCheckpoint('best_baseline_model.h5', 
                             save_best_only=True, 
                             save_weights_only=True,
                             monitor='val_loss', 
                             mode='min', verbose=1)

# Assuming 'model' has been defined and compiled (for example, using a function like create_model())
# Run the training process
history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=[model_save, early_stop, reduce_lr]
)




val_imgs, val_labels = next(iter(validation_generator))
predictions = model.predict(val_imgs)
predicted_classes = np.argmax(predictions, axis=1)  # Convert probabilities to class labels

# True labels are required in the same order as predictions
true_classes = val_labels

# Generate the classification report
report = classification_report(true_classes, predicted_classes, target_names=label_names)
print(report)









preds = []

for image_id in test_df.image_id:
    image = Image.open(os.path.join(Dir,  "test_images", image_id))
    image = image.resize((target_size_dim,target_size_dim))
    image = np.expand_dims(image, axis = 0)
    preds.append(np.argmax(model.predict(image)))

test_df['label'] = preds
test_df

test_df.to_csv('submission.csv', index = False)
