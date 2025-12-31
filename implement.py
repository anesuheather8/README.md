import cv2
from tensorflow.keras.models import load_model
import legacy_sm_saving_lib
import numpy as np

#load the trained model
model = load_model('C:\\Users\\Pana\\anaconda3\\envs\\sign_language_model.h5')

#initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    #capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    #convert the frame to RGB, resize and normalize
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128)) / 255.0 
    image = np.expand_dims(image, axis=0)

    #make prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)

    #display the predicted sign on the frame
    cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    #display the resulting frame
    cv2.imshow('Camera Feed', frame)

    #break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
#release the capture and close windows 
cap.release()
cv2.destroyAllWindows()