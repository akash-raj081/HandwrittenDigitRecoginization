import numpy as np
import cv2  # For image processing
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('handwritten_digit_model.keras')

# Function to preprocess the image (resize, normalize, and reshape)
def preprocess_image(image_path, image_size=28):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to the target size (e.g., 28x28 for MNIST-like images)
    img = cv2.resize(img, (image_size, image_size))
    
    # Normalize the pixel values (0-255 to 0-1)
    img = img.astype('float32') / 255.0
    
    # Reshape the image to match the model's input shape
    img = img.reshape((1, image_size, image_size, 1))  # Adding batch dimension and channel
    return img

# Test the model on a sample image
image_path = 'test_image.png'  # Replace with the actual image file path
processed_image = preprocess_image(image_path)

# Predict the label using the model
prediction = model.predict(processed_image)
predicted_class = np.argmax(prediction, axis=1)  # Get the class with the highest probability

# Print the prediction result
print(f"Predicted Label: {predicted_class[0]}")  # Directly print the predicted label

# Optionally, display the image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title(f"Predicted Label: {predicted_class[0]}")
plt.axis('off')
plt.show()