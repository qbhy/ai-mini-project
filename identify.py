import cv2
import face_recognition
import os
import numpy as np

# 定义人脸数据库存储路径
face_db_path = './face_db'

# 创建数据库文件夹
if not os.path.exists(face_db_path):
    os.makedirs(face_db_path)

# 加载人脸数据库
known_face_encodings = []
known_face_names = []

for file in os.listdir(face_db_path):
    if file.endswith('.npy'):
        face_encoding = np.load(os.path.join(face_db_path, file))
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(file)[0])


def detect_faces_in_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Create an output movie file (make sure resolution/frame rate matches input video!)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    detected_faces = []

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            detected_faces.append(name)

    video_capture.release()
    return detected_faces


def detect_faces_from_image(image_path):
    detected_faces = []

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)

    # Find all the faces and face encodings in the current frame of video
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        detected_faces.append(name)

    return detected_faces


result = detect_faces_in_video("saitama.mp4")
print(result)
