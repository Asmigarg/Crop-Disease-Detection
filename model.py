import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2

rgb_model = tf.keras.models.load_model('model/plant_model_rgb.keras')
hsv_model = tf.keras.models.load_model('model/plant_model_hsv.keras')

class_names = [
    'Apple_Apple_Scab', 'Apple_Black_Rot', 'Apple_Cedar_Apple_Rust', 'Apple_Healthy', 'Bell_Pepper_Bacterial_Spot', 'Bell_Pepper_Healthy', 'Cherry_Healthy', 'Cherry_Powdery_Mildew', 'Corn_Cercospora_Leaf_Spot', 'Corn_Common_Rust', 'Corn_Healthy', 'Corn_Northern_Leaf_Blight', 'Grape_Black_Rot', 'Grape_Esca_Black_Measles', 'Grape_Healthy', 'Grape_Leaf_Blight', 'Peach_Bacterial_Spot', 'Peach_Healthy', 'Potato_Early_Blight', 'Potato_Healthy', 'Potato_Late_Blight', 'Strawberry_Healthy', 'Strawberry_Leaf_Scorch', 'Tomato_Bacterial_Spot', 'Tomato_Early_Blight', 'Tomato_Healthy', 'Tomato_Late_Blight', 'Tomato_Septoria_Leaf_Spot', 'Tomato_Yellow_Leaf_Curl_Virus'
]

def load_and_preprocess_rgb_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def load_and_preprocess_hsv_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array.astype(np.uint8)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict(img_path):
    rgb_img_array = load_and_preprocess_rgb_image(img_path)
    hsv_img_array = load_and_preprocess_hsv_image(img_path)
    
    rgb_pred = rgb_model.predict(rgb_img_array)[0]
    hsv_pred = hsv_model.predict(hsv_img_array)[0]

    final_pred = (0.25 * rgb_pred) + (0.75 * hsv_pred)

    predicted_class_index = np.argmax(final_pred)
    predicted_class = class_names[predicted_class_index]
    confidence_score = final_pred[predicted_class_index]

    if (confidence_score < 0.9):
        return "cant_predict"
    else:
        return f"{predicted_class}+{confidence_score}"