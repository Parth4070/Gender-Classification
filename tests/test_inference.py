from PIL import Image

from src.predict import predict_image


IMAGE_PATH = "dataset/test/women/121.jpg"


image = Image.open(IMAGE_PATH)

prediction, confidence = predict_image(image)

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.4f}")