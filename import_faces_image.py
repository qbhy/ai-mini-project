import face_recognition
import os
import numpy as np

# 定义人脸数据库存储路径
face_db_path = './face_db'

# 创建数据库文件夹
if not os.path.exists(face_db_path):
    os.makedirs(face_db_path)


def capture_face_from_image(image_path, name):
    # 加载图像
    image = face_recognition.load_image_file(image_path)

    # 检测人脸
    face_locations = face_recognition.face_locations(image)
    if face_locations:
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        np.save(os.path.join(face_db_path, f'{name}.npy'), face_encoding)
        print(f"Face data for {name} saved successfully.")
    else:
        print("No face detected in the image.")


# this is my picture
faces = [
    ["saitama", "saitama.jpg"],
]

for name, imagePath in faces:
    capture_face_from_image(imagePath, name)
