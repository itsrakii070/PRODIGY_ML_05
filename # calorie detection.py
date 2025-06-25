import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Calorie map (per 100g)
calorie_map = {
    "apple": 52,
    "banana": 89,
    "pizza": 266,
    "burger": 295,
    "fries": 312,
    "salad": 33
}

# Load model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
output = Dense(len(calorie_map), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
# Load trained weights here if available: model.load_weights('food_model.h5')

# Sample labels (same order as model output)
labels = list(calorie_map.keys())

def predict_food(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    food_name = labels[class_idx]
    calories = calorie_map[food_name]

    print(f"Predicted: {food_name}")
    print(f"Estimated Calories: {calories} kcal per 100g")

# Example usage
predict_food("example_food.jpg")
