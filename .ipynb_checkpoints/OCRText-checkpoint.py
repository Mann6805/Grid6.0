import cv2
import easyocr

# Initialize video capture
video = cv2.VideoCapture(0)
video.set(3, 640)  # Set video width
video.set(4, 480)  # Set video height

# Initialize EasyOCR reader for Hindi and English
reader = easyocr.Reader(['hi', 'en'])

while True:
    ret, frame = video.read()
    # feed = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect

    if ret:
        # Perform OCR on the frame
        result = reader.readtext(frame)
        
        # Loop through detected text
        for detection in result:
            # Bounding box coordinates
            top_left = tuple([int(val) for val in detection[0][0]])
            bottom_right = tuple([int(val) for val in detection[0][2]])
            
            # Draw rectangle around detected text
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            
            # Display the detected text above the rectangle
            text = detection[1]
            cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display the video feed with rectangles and text
        cv2.imshow("Video", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object and close windows
video.release()
cv2.destroyAllWindows()
