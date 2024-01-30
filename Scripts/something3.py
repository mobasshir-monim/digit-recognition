import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained digit recognition model
model = load_model('mnist_digit_recognition_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply image processing techniques specific to LED display digits
    # You may need to adjust these based on your specific case
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    img = cv2.medianBlur(img,3)
    
    return img

def extract_digits(image):
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        # Filter out small or too elongated contours 
        if 10 < w < 1000 and 1.5 > aspect_ratio > 0.5:
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

def read_digits_from_led_display(image_path):
    processed_image = preprocess_image(image_path)
    digits = extract_digits(processed_image)
    recognized_digits = recognize_digits(digits)
    return recognized_digits

# Example usage
image_path = 'M:/a25e02b9-bd79-4c7a-acb6-f198aca7cf10.jpg'
predicted_digits = read_digits_from_led_display(image_path)
print(f'The recognized digits from the LED display are: {predicted_digits}')
