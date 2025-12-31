import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Load your trained model (make sure to specify the correct path)
model = torch.load('path_to_your_model.pth')
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to the input size your model expects
    transforms.ToTensor(),
])

# Initialize the camera
cap = cv2.VideoCapture(0)  # Change 0 if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply transformations
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
    # Map predicted label to actual text (you may need to adjust this)
    predicted_label = predicted.item()  # Get the predicted label
    # Convert label to text using your label mapping
    label_text = "Predicted: " + str(predicted_label)  # Replace with your mapping logic

    # Display the resulting frame
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Sign Language Recognition', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
