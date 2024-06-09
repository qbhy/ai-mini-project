import cv2
import numpy as np
import time
import paho.mqtt.client as mqtt

# Load the object detection model (e.g. YOLOv9)
net = cv2.dnn.readNet("yolov9.weights", "yolov9.cfg")

# Define the classes of interest (gym equipment)
classes = ["treadmill", "weight_machine", "bench_press", "elliptical"]

# Load the facial recognition model (e.g. FaceNet)
face_net = cv2.dnn.readNet("facenet_weights", "facenet_cfg")

# Set up the camera
cap = cv2.VideoCapture(0)

# Initialize variables for tracking equipment usage
equipment_start_time = {}
equipment_user = {}

# Set up the MQTT client
broker_address = "broker.hivemq.com"
port = 1883
client = mqtt.Client("exer1")
client.connect(broker_address, port)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Pre-process the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    # Run object detection
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists to store detected objects and their bounding boxes
    objects = []
    boxes = []

    # Iterate through the detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id in [classes.index(c) for c in classes]:
                # Get the bounding box coordinates
                x, y, w, h = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                objects.append(classes[class_id])
                boxes.append((x, y, w, h))

    # Draw bounding boxes around detected objects
    for i, obj in enumerate(objects):
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, obj, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check if a person is using the equipment
        if obj in equipment_start_time and time.time() - equipment_start_time[obj] > 60:
            # Log the usage data to the member's account
            print(f"Member {equipment_user[obj]} used {obj} for {time.time() - equipment_start_time[obj]} seconds")
            # Reset the start time and user for the equipment
            equipment_start_time[obj] = 0
            equipment_user[obj] = ""

        # Check if a person is approaching the equipment
        if confidence > 0.7 and obj in equipment_start_time and equipment_start_time[obj] == 0:
            # Identify the user based on facial recognition
            face_roi = frame[int(y):int(y + h), int(x):int(x + w)]
            face_blob = cv2.dnn.blobFromImage(face_roi, 1 / 255, (160, 160), [0, 0, 0], 1, crop=False)
            face_net.setInput(face_blob)
            face_outs = face_net.forward(face_net.getUnconnectedOutLayersNames())
            face_id = np.argmax(face_outs)
            # Update the user and start time for the equipment
            equipment_user[obj] = face_id
            equipment_start_time[obj] = time.time()

    # Publish the detection results to the MQTT broker
    client.publish("estatus", objects)

    # Display the output
    cv2.imshow("Gym Equipment Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
client.disconnect()