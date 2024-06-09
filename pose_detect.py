import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from datetime import datetime
from ultralytics import YOLO

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

model = YOLO("best.pt")


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return round(angle, 2)


def show_loading_label():
    global loading_label
    loading_label = tk.Label(root, text="Loading...", font=("Arial", 12))
    loading_label.pack(pady=20)


def hide_loading_label():
    loading_label.destroy()


times = {}


def recoding(source, actions):
    global times
    show_loading_label()  # Show the loading text
    root.update()  # Update the GUI to display the loading text

    cap = cv2.VideoCapture(source)
    hide_loading_label()
    root.update()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frameIndex = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            ++frameIndex
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            if results.pose_landmarks:

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )  # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                detection = None
                if frameIndex % 3 == 0:
                    detection = model(image)[0]

                actionsCount = {}
                for name, action in actions:
                    result = action(results, detection)
                    if isinstance(result, list):
                        [count, image] = result
                    else:
                        count = result

                    if count > 0:
                        actionsCount[name] = count
                    if name not in times:
                        times[name] = datetime.now()

                    if count >= 1:
                        # 计算两个时间点之间的时间差
                        time_difference = datetime.now() - times[name]

                        # 获取时间差的天、秒数
                        days = time_difference.days
                        seconds = time_difference.seconds

                        # 转换为小时、分钟和秒
                        hours = days * 24 + seconds // 3600
                        minutes = (seconds % 3600) // 60
                        seconds = seconds % 60

                        # 构建时间差的字符串
                        timeString = ""
                        if hours > 0:
                            timeString += f"{hours} h, "
                        if minutes > 0 or hours > 0:
                            timeString += f"{minutes} m, "
                        timeString += f"{seconds} s"
                    else:
                        timeString = '0s'
                    print(f"Action: {name}, count: {count} \t time: {timeString}")

                    if count > 0:
                        cv2.putText(image, f'{name}    {timeString}',
                                    (10, len(actionsCount) * 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('Mediapipe Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


# 初始化全局变量
shoulder_press_counter = 0
shoulder_press_stage = None


def shoulder_press_action(results, detection):
    global shoulder_press_counter, shoulder_press_stage
    image = None
    if detection is not None:
        image = detection.plot()
        if "Shoulder Press" not in [model.names[int(box.cls)] for box in detection.boxes]:
            return shoulder_press_counter

    landmarks = results.pose_landmarks.landmark

    # 获取左肩、左肘、左手腕的关键点坐标
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    # 获取右肩、右肘、右手腕的关键点坐标
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    # 计算左臂和右臂的角度
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # 检测肩推状态
    if left_angle > 160 and right_angle > 160:
        shoulder_press_stage = "up"
    if (left_angle < 90 or right_angle < 90) and shoulder_press_stage == "up":
        shoulder_press_stage = "down"
        shoulder_press_counter += 1
        print(f"Shoulder Press Count: {shoulder_press_counter}")

    return [shoulder_press_counter, image] if detection is not None else shoulder_press_counter


# 初始化全局变量
dumbbell_curl_counter = 0
dumbbell_curl_stage = None


def dumbbell_curl_action(results, detection):
    global dumbbell_curl_counter, dumbbell_curl_stage

    image = None
    if detection is not None:
        image = detection.plot()
        if "Dumbbell" not in [model.names[int(box.cls)] for box in detection.boxes]:
            return shoulder_press_counter

    landmarks = results.pose_landmarks.landmark

    # 获取左肩、左肘、左手腕的关键点坐标
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    # 获取右肩、右肘、右手腕的关键点坐标
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    # 计算左臂的角度
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    # 计算右臂的角度
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # 检测哑铃弯举状态
    if left_angle > 145 or right_angle > 145:
        dumbbell_curl_stage = "down"
    if (left_angle < 90 or right_angle < 90) and dumbbell_curl_stage == "down":
        dumbbell_curl_stage = "up"
        dumbbell_curl_counter += 1
        print(f"Dumbbell Curl Count: {dumbbell_curl_counter}")

    return [dumbbell_curl_counter, image] if detection is not None else dumbbell_curl_counter


# 初始化全局变量
lateral_raise_counter = 0
lateral_raise_stage = None


def lateral_raise_action(results, detection):
    global lateral_raise_counter, lateral_raise_stage

    landmarks = results.pose_landmarks.landmark

    # 获取左肩、左肘、左手腕的关键点坐标
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    # 获取右肩、右肘、右手腕的关键点坐标
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    # 获取左髋、左膝的关键点坐标
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

    # 获取右髋、右膝的关键点坐标
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

    # 计算左臂的角度
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    # 计算右臂的角度
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # 计算左腿的角度
    left_leg_angle = calculate_angle(left_hip, left_knee, [left_hip[0], left_hip[1] + 0.1])

    # 计算右腿的角度
    right_leg_angle = calculate_angle(right_hip, right_knee, [right_hip[0], right_hip[1] + 0.1])

    # 检测侧举状态
    if left_arm_angle > 160 and right_arm_angle > 160 and (left_leg_angle < 165 or right_leg_angle < 165):
        lateral_raise_stage = "down"
        print("Arms down")

    if left_arm_angle < 30 and right_arm_angle < 30 and lateral_raise_stage == "down" and (
            left_leg_angle < 165 or right_leg_angle < 165):
        lateral_raise_stage = "up"
        lateral_raise_counter += 1
        print(lateral_raise_counter)

    return lateral_raise_counter


# 初始化全局变量
tricep_extension_counter = 0
tricep_extension_stage = None


def tricep_ext_action(results, detection):
    global tricep_extension_counter, tricep_extension_stage

    landmarks = results.pose_landmarks.landmark

    # 获取左肩、左肘、左手腕的关键点坐标
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    # 获取右肩、右肘、右手腕的关键点坐标
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    # 计算左臂的角度
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    # 计算右臂的角度
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # 检测三头肌伸展状态
    if left_angle > 160 and right_angle > 160:
        tricep_extension_stage = "down"
        print("Arms down")

    if left_angle < 30 and right_angle < 30 and tricep_extension_stage == "down":
        tricep_extension_stage = "up"
        tricep_extension_counter += 1
        print(tricep_extension_counter)

    return tricep_extension_counter


# Squat counter variables
squat_counter = 0
squat_stage = None


def squats_action(results, detection):
    global squat_counter, squat_stage

    landmarks = results.pose_landmarks.landmark

    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

    left_angle = calculate_angle(left_ankle, left_knee, left_hip)

    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

    right_angle = calculate_angle(right_ankle, right_knee, right_hip)

    if left_angle < 90 and right_angle < 90:
        squat_stage = "compressed"
        print("compressed")

    if left_angle > 165 and right_angle > 165 and squat_stage == "compressed":
        squat_stage = "extended"
        squat_counter += 1
        print(squat_counter)

    return squat_counter


# 初始化全局变量
leg_extension_counter = 0
leg_extension_stage = None


def leg_ext_action(results, detection):
    global leg_extension_counter, leg_extension_stage

    landmarks = results.pose_landmarks.landmark

    # 获取左髋、左膝、左脚踝的关键点坐标
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    # 获取右髋、右膝、右脚踝的关键点坐标
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # 计算左腿的角度
    left_angle = calculate_angle(left_hip, left_knee, left_ankle)

    # 计算右腿的角度
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # 检测腿部伸展状态
    if left_angle > 160 and right_angle > 160:
        leg_extension_stage = "down"
        print("Legs down")

    if left_angle < 30 and right_angle < 30 and leg_extension_stage == "down":
        leg_extension_stage = "up"
        leg_extension_counter += 1
        print(leg_extension_counter)

    return leg_extension_counter


# 初始化全局变量
jump_rope_counter = 0
jump_rope_stage = None


def jump_rope_action(results, detection):
    global jump_rope_counter, jump_rope_stage

    landmarks = results.pose_landmarks.landmark

    # 获取左肩、左肘、左手腕的关键点坐标
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    # 获取右肩、右肘、右手腕的关键点坐标
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    # 获取左髋、左膝、左踝的关键点坐标
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    # 获取右髋、右膝、右踝的关键点坐标
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # 计算左臂的角度
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    # 计算右臂的角度
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # 计算左腿的角度
    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)

    # 计算右腿的角度
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # 检测跳绳状态
    if left_leg_angle > 170 and right_leg_angle > 170 and 80 < left_arm_angle < 130 and 80 < right_arm_angle < 130:
        jump_rope_stage = "landed"
        print("Landed")

    if left_leg_angle < 160 and right_leg_angle < 160 and jump_rope_stage == "landed" and 80 < left_arm_angle < 130 and (
            80 < right_arm_angle < 130):
        jump_rope_stage = "jumped"
        jump_rope_counter += 1
        print(f"Jump Rope Count: {jump_rope_counter}")

    return jump_rope_counter


# 初始化全局变量
high_knees_counter = 0
high_knees_stage = None


def high_knees_action(results, detection):
    global high_knees_counter, high_knees_stage

    landmarks = results.pose_landmarks.landmark

    # 获取左髋、左膝的关键点坐标
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

    # 获取右髋、右膝的关键点坐标
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

    # 设置高抬腿检测阈值
    knee_hip_diff_threshold = 0.15  # 根据需求调整该值

    # 检测左腿的高抬腿状态
    if left_knee[1] < left_hip[1] - knee_hip_diff_threshold and right_knee[1] > right_hip[1] + knee_hip_diff_threshold:
        if high_knees_stage == "down":
            high_knees_stage = "up"
            high_knees_counter += 1
            print("Left leg up")
            print(high_knees_counter)
    if left_knee[1] > left_hip[1] and high_knees_stage == "up":
        high_knees_stage = "down"

    # 检测右腿的高抬腿状态
    if right_knee[1] < right_hip[1] - knee_hip_diff_threshold and left_knee[1] > left_hip[1] + knee_hip_diff_threshold:
        if high_knees_stage == "down":
            high_knees_stage = "up"
            high_knees_counter += 1
            print("Right leg up")
            print(high_knees_counter)
    if right_knee[1] > right_hip[1] and high_knees_stage == "up":
        high_knees_stage = "down"

    return high_knees_counter


# 初始化全局变量
pull_ups_counter = 0
pull_ups_stage = None


def pull_ups_action(results, detection):
    global pull_ups_counter, pull_ups_stage

    landmarks = results.pose_landmarks.landmark

    # 获取左肩、左肘、左手腕的关键点坐标
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    # 获取右肩、右肘、右手腕的关键点坐标
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    # 计算左臂的角度
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    # 计算右臂的角度
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # 检测引体向上状态
    if left_angle > 160 and right_angle > 160:
        pull_ups_stage = "down"
        print("Arms down")

    if left_angle < 30 and right_angle < 30 and pull_ups_stage == "down":
        pull_ups_stage = "up"
        pull_ups_counter += 1
        print(pull_ups_counter)

    return pull_ups_counter


# 初始化全局变量
leg_press_counter = 0
leg_press_stage = None


def leg_press_action(results, detection):
    global leg_press_counter, leg_press_stage
    image = None
    # 渲染检测结果
    # if detection is None:
        # image = detection.plot()
        # if "Leg Press" not in [model.names[int(box.cls)] for box in detection.boxes]:
        #     print("Leg Press machine is not detected in labels.")
        #     return leg_press_counter

    landmarks = results.pose_landmarks.landmark

    # 获取左髋、左膝、左踝的关键点坐标
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    # 获取右髋、右膝、右踝的关键点坐标
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # 计算左腿的角度
    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)

    # 计算右腿的角度
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # 检测腿推举状态
    if left_leg_angle > 160 and right_leg_angle > 160:
        leg_press_stage = "down"
        print("Legs down")

    if left_leg_angle < 90 and right_leg_angle < 90 and leg_press_stage == "down":
        leg_press_stage = "up"
        leg_press_counter += 1
        print(f"Leg Press Count: {leg_press_counter}")

    return [leg_press_counter, image] if detection is not None else leg_press_counter


def main():
    global root
    # Create the main window
    root = tk.Tk()
    root.title("Workout Menu")

    root.geometry("400x750")

    arms_label = tk.Label(root, text="Arms", font=("Helvetica", 15, "bold"))
    arms_label.pack(pady=10)

    video = 'demo/leg_press.mp4'

    button = tk.Button(root, text="start", command=lambda: recoding(video, [
        ["squats_action", squats_action],
        ["shoulder_press_action", shoulder_press_action],
        ["pull_ups_action", pull_ups_action],
        ["dumbbell_curl_action", dumbbell_curl_action],
        ["jump_rope_action", jump_rope_action],
        ["lateral_raise_action", lateral_raise_action],
        ["tricep_ext_action", tricep_ext_action],
        ["leg_ext_action", leg_ext_action],
        ["high_knees_action", high_knees_action],
    ]))
    button.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
