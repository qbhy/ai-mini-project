import os
import cv2
import face_recognition
import numpy as np


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def extract_and_save_face_embeddings(known_faces, output_dir):
    ensure_directory(output_dir)

    for name, video_path in known_faces:
        video_capture = cv2.VideoCapture(video_path)

        face_encodings = []

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            face_locations = face_recognition.face_locations(rgb_frame)

            try:
                face_encodings_in_frame = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings_in_frame:
                    face_encodings.append(face_encodings_in_frame[0])  # Assume one face per frame
            except Exception as e:
                print(f"Error processing frame: {e}")

        video_capture.release()

        if face_encodings:
            # Average the face encodings to get a single representation of the face
            mean_face_encoding = np.mean(face_encodings, axis=0)
            np.save(os.path.join(output_dir, f"{name}.npy"), mean_face_encoding)
            print(f"Saved {name}'s face encoding to {output_dir}")
        else:
            print(f"No faces found in video {video_path}")


# 示例用法
known_faces = [
    # ['saitama', 'saitama.mp4'],
    # ['jin', 'jin.mp4'],
    ['yuan', 'yuan.mp4'],
]
output_directory = 'face_db'
extract_and_save_face_embeddings(known_faces, output_directory)