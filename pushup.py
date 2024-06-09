import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化视频捕获
cap = cv2.VideoCapture('pushup.mp4')  # 替换为你的视频文件路径

# 定义俯卧撑检测的状态和计数器
pushup_state = None
pushup_counter = 0


def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Second point (center point)
    c = np.array(c)  # Third point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # 进行姿态检测
    results = pose.process(image)

    # 绘制姿态标记
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        # 获取必要的关键点
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # 计算肘关节的角度
        angle = calculate_angle(shoulder, elbow, wrist)

        # 显示角度
        cv2.putText(image, str(angle),
                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 检测俯卧撑状态
        if angle > 160:
            if pushup_state == 'down':
                pushup_counter += 1
                pushup_state = 'up'
        if angle < 30:
            pushup_state = 'down'

        # 显示俯卧撑计数
        cv2.putText(image, 'Pushups: ' + str(pushup_counter), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    except:
        pass

    # 显示结果图像
    cv2.imshow('Pushup Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()