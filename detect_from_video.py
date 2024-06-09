import os
import cv2
import face_recognition
import numpy as np


def load_known_faces(directory):
    known_face_encodings = []
    known_face_names = []

    for file in os.listdir(directory):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]
            encoding = np.load(os.path.join(directory, file))
            known_face_encodings.append(encoding)
            known_face_names.append(name)

    return known_face_encodings, known_face_names


def recognize_faces_in_video(video_path, known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.3)
            name = "unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# 示例用法
known_face_encodings, known_face_names = load_known_faces('face_db')

# 要检测人脸的视频文件
video_path = 'saitama.mp4'
recognize_faces_in_video(video_path, known_face_encodings, known_face_names)
