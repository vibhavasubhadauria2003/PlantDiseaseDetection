
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

CLASSES = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_healthy'
]

IMG_SIZE = (224, 224)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def predict_image(image_path, model_type="mobilenet"):

    if model_type == "mobilenet":
        model_path = os.path.join(
            PROJECT_ROOT, "models", "mobilenetv2.keras"
        )
    elif model_type == "custom_cnn":
        model_path = os.path.join(
            PROJECT_ROOT, "models", "custom_cnn.keras"
        )
    else:
        raise ValueError("Invalid model type")

    model = tf.keras.models.load_model(model_path)

    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if model_type == "mobilenet":
        processed_image = preprocess_input(img_array.copy())
    else:
        processed_image = img_array / 255.0

    predictions = model.predict(processed_image, verbose=0)

    predicted_index = np.argmax(predictions[0])

    return {
        "disease": CLASSES[predicted_index],
        "confidence": float(predictions[0][predicted_index] * 100),
        "probabilities": predictions[0].tolist()
    }


if __name__ == "__main__":

    image_path = input("Enter leaf image path: ")

    result = predict_image(image_path)

    print("\nMobileNetV2 Prediction Result")
    print("----------------------")
    print("Disease:", result["disease"])
    print("Confidence:", round(result["confidence"], 2), "%")
    
    result2 = predict_image(image_path, model_type="custom_cnn")
    
    print("\n Custom CNN Prediction Result")
    print("----------------------")
    print("Disease:", result2["disease"])
    print("Confidence:", round(result2["confidence"], 2), "%")
