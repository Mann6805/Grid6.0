import os
import pandas as pd
import numpy as np
import cv2
import easyocr
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128, 128)

# class_labels = [' apple', ' instant_noodle', ' juice', ' orange', ' sandwich'] #Grocery
class_labels = [' AfterShave', ' FabricBrightener', ' FaceMask', ' Handwash', ' Lotion', ' Napkin', ' Oil', ' PetroliumJelly', ' Powder', ' Soap', ' Sweetner', ' Toothpaste'] #FMCG

# Function to preprocess the input frame
def preprocess_frame(frame):
    # Resize to the model's input size
    resized_frame = cv2.resize(frame, IMG_SIZE)
    # Normalize the pixel values
    normalized_frame = resized_frame / 255.0
    # Add batch dimension
    return np.expand_dims(normalized_frame, axis=0)

# Initialize video capture
video = cv2.VideoCapture(0)
video.set(3, 320)  # Reduce video width (for lower resolution)
video.set(4, 240)  # Reduce video height (for lower resolution)

model = tf.keras.models.load_model('FMCG.h5')  # Adjust this to your model path

# Initialize EasyOCR reader for only English (this reduces overhead)
reader = easyocr.Reader(['en','hi'], gpu=False)  # Disable GPU to save memory

# Run OCR on every Nth frame
frame_skip = 20  # Perform OCR every 10th frame to speed things up
frame_count = 0

# List to store detected texts and bounding boxes
stored_results = []

while True:
    ret, frame = video.read()
    if ret:
        # feed = cv2.flip(frame, 1)  # Flip frame horizontally

        # Only run OCR on every Nth frame to save processing time
        if frame_count % frame_skip == 0:
            result = reader.readtext(frame)

            # Store new detections in the list
            stored_results = [
                {
                    "top_left": tuple([int(val) for val in detection[0][0]]),
                    "bottom_right": tuple([int(val) for val in detection[0][2]]),
                    "text": detection[1]
                }
                for detection in result
            ]

        input_data = preprocess_frame(frame)

        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions, axis=-1)
        predicted_probability = np.max(predictions, axis=-1)

        if predicted_probability[0] > 0.5:
            predicted_class_name = class_labels[predicted_class[0]]
            cv2.putText(frame, f'Predicted Class: {predicted_class_name}', (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Loop through stored detections to draw bounding boxes and text
        for stored in stored_results:
            # Draw rectangle around detected text
            cv2.rectangle(frame, stored["top_left"], stored["bottom_right"], (0, 255, 0), 2)

            # Display the detected text above the rectangle
            cv2.putText(frame, stored["text"], (stored["top_left"][0], stored["top_left"][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
           

        # Display the video frame with rectangles and text
        cv2.imshow("Video", frame)

        frame_count += 1

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object and close windows
video.release()
cv2.destroyAllWindows()