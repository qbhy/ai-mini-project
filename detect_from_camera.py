import face_recognition
import cv2
import numpy as np
import os

# 加载人脸数据

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

# 创建窗口
cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 按钮坐标和尺寸
button_x, button_y, button_w, button_h = 20, 20, 100, 40

# 按钮文本
button_text = 'Start'

# 按钮颜色
button_color = (255, 255, 255)

# 按钮文本颜色
text_color = (0, 0, 0)

# 是否启动人脸识别
start_recognition = False


def on_button_click(event, x, y, flags, param):
    global start_recognition
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h:
            start_recognition = not start_recognition


# 设置鼠标事件处理器
cv2.setMouseCallback('Face Recognition', on_button_click)

while True:
    # 捕获视频流中的一帧
    ret, frame = cap.read()

    if not ret:
        break

    # 如果启动了人脸识别
    if start_recognition:
        # 将图像转换为RGB颜色格式
        rgb_frame = frame[:, :, ::-1]

        # 检测人脸位置
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # 绘制按钮
    cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), button_color, -1)
    cv2.putText(frame, button_text, (button_x + 10, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # 在窗口中显示图像
    cv2.imshow('Face Recognition', frame)

    # 检测鼠标事件
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
