import tensorflow as tf

model = tf.keras.models.load_model("models/best_tomato_model.keras")

print("Model loaded successfully")
print(model.input_shape)
print(model.output_shape)