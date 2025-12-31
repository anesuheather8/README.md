import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('sign_language_model.h5')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB, resize, and normalize
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128)) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)

    # Display the predicted sign on the frame
    cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
