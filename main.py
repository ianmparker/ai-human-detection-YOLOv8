# Import necessary libraries
import random
import cv2
import numpy as np
from ultralytics import YOLO

# Open the file containing class names in read mode
my_file = open("utils\coco.txt", "r")

# Read the file content
data = my_file.read()

# Split the text into a list of class names. The split occurs at every newline ('\n')
class_list = data.split("\n")

# Close the file
my_file.close()

# Generate random colors for each class. This will be used to color the bounding boxes for detected objects
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load a pretrained YOLOv8n model from the specified path
model = YOLO("weights/yolov8n.pt", "v8")

# Define the width and height to resize video frames. Smaller frames can speed up the processing
frame_wid = 640
frame_hyt = 480

# Open the video file or camera stream
cap = cv2.VideoCapture("NSBE 2024 Conference.mov")

# Start a loop that will run until the video ends or 'q' is pressed
while True:
    # Capture each frame from the video
    ret, frame = cap.read()

    # If the frame was not captured correctly, print an error message and break the loop
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Use the model to predict objects in the frame. The confidence threshold is set to 0.45
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Convert the tensor array of detection parameters to a numpy array
    DP = detect_params[0].numpy()

    # If any objects were detected in the frame
    if len(DP) != 0:
        # Loop over each detected object
        for i in range(len(detect_params[0])):
            # Get the bounding box, class ID, and confidence of the current object
            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            # Draw a rectangle on the frame around the detected object
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display the class name and confidence on the frame
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Resize and display the frame with detected objects and annotations
    frame = cv2.resize(frame, (800, 600)) 
    cv2.imshow("ObjectDetection", frame)

    # If 'q' is pressed on the keyboard, break the loop
    if cv2.waitKey(1) == ord("q"):
        break

# When the loop is done, release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
