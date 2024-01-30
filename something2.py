import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained digit recognition model
model = load_model('mnist_digit_recognition_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (300, 150))  # Resize for better digit detection
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return img

def extract_digits(image):
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        digit = image[y:y+h, x:x+w]
        digit = cv2.resize(digit, (28, 28))
        digit = digit.reshape(1, 28, 28, 1).astype('float32') / 255.0
        digit_images.append((x, digit))
    
    digit_images.sort(key=lambda x: x[0])  # Sort digits by x-coordinate
    return [digit for _, digit in digit_images]

def recognize_digits(digits):
    recognized_digits = []
    for digit in digits:
        prediction = model.predict(digit)
        recognized_digits.append(np.argmax(prediction))
    return recognized_digits

def read_multiple_digits(image_path):
    processed_image = preprocess_image(image_path)
    digits = extract_digits(processed_image)
    recognized_digits = recognize_digits(digits)
    return recognized_digits

# Example usage
image_path = 'M:/Screenshot 2024-01-29 225959.png'
predicted_digits = read_multiple_digits(image_path)
print(f'The recognized digits are: {predicted_digits}')
