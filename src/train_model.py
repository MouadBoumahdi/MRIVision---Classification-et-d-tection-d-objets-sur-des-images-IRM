import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from preprocessing import train_ds, val_ds, test_ds


base_model = MobileNetV2(
    weights = "imagenet",
    include_top=False,
    input_shape=(224,224,3),
)


base_model.summary()

base_model.trainable = False
inputs = layers.Input(shape=(224,224,3))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(4, activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)


model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience=3,
    restore_best_weights=True
)


# history = model.fit(
#     train_ds,
#     validation_data = val_ds,
#     epochs=10,
#     callbacks=[early_stopping]
# )

model.save("mobilenetv2.keras")


test_loss, test_accurcay = model.evaluate(test_ds)
print("Test accuracy: ",test_accurcay)
print("Test Loss: ",test_loss)


