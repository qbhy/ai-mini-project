import os
from ultralytics import YOLO
import cv2
from collections import defaultdict


def process_images(input_folder, output_folder, model):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 初始化标签统计字典
    label_stats = defaultdict(int)

    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # 进行检测
            results = model(image)

            for result in results:
                # 渲染检测结果
                annotated_image = result.plot()

                # 统计标签数量
                for box in result.boxes:
                    label = model.names[int(box.cls)]
                    label_stats[label] += 1

                # 保存结果图片
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, annotated_image)
                print(f"Processed and saved: {output_path}")

    # 输出统计信息
    print("\nDetection Statistics:")
    for label, count in label_stats.items():
        print(f"{label}: {count}")


# 示例使用
model_path = 'best.pt'  # 替换为你的模型路径
input_folder = 'gym_equipment/test/images'  # 替换为你的输入文件夹路径
output_folder = 'gym_equipment_results'  # 输出文件夹

# 加载模型
model = YOLO(model_path)

# 处理图片并保存结果
process_images(input_folder, output_folder, model)
