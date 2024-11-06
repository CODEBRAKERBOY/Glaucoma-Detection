import tensorflow as tf
from vit_keras import vit
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Set global variables
IMG_SIZE = (128, 128)
SEED = 42
BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE

# Data Processing Functions
def img_preprocessing(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size=IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augmentation(image, label):
    img = tf.image.random_flip_left_right(image, seed=SEED)
    img = tf.image.random_brightness(img, max_delta=0.1, seed=SEED)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2, seed=SEED)
    return img, label

# Prepare Dataset
def prepare_dataset(image_paths, labels, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(img_preprocessing, num_parallel_calls=AUTO)
    dataset = dataset.map(augmentation, num_parallel_calls=AUTO)
    dataset = dataset.shuffle(buffer_size=len(image_paths), seed=SEED)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset

# Assuming train_df, val_df, test_df are loaded with image paths and labels
train_dataset = prepare_dataset(train_df['image'].values, train_df['label'].values)
val_dataset = prepare_dataset(val_df['image'].values, val_df['label'].values)
test_dataset = prepare_dataset(test_df['image'].values, test_df['label'].values)

# Vision Transformer (ViT) Model Definition
def create_vit_model():
    vit_model = vit.vit_b16(image_size=IMG_SIZE, activation='softmax', pretrained=True, 
                            include_top=False, pretrained_top=False, classes=2)
    inp = Input(shape=(*IMG_SIZE, 3))
    vit_output = vit_model(inp)
    x = Flatten()(vit_output)
    x = Dense(256, activation='gelu')(x)
    x = Dense(64, activation='gelu')(x)
    x = Dense(32, activation='gelu')(x)
    out = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=inp, outputs=out)
    return model

# Compile the Model
model = create_vit_model()
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Early Stopping Callback
early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5)

# Train the Model
history = model.fit(train_dataset, epochs=30, batch_size=BATCH_SIZE,
                    validation_data=val_dataset, callbacks=[early_stopping])

# Save the model
model.save('/content/drive/My Drive/DATASET/trained_vit_model.h5')
