import numpy as np
from keras import models
import cv2
from scipy.spatial.distance import euclidean

# 加载预训练模型
model = models.load_model('facenet_keras.h5')


# 预处理图像函数
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    return np.expand_dims(img, axis=0)


# 加载已知人脸的特征向量
def load_known_faces(known_faces):
    known_embeddings = {}
    for name, image_path in known_faces:
        image = preprocess_image(image_path)
        embedding = model.predict(image)
        known_embeddings[name] = embedding
    return known_embeddings


# 预测图像中的人脸
def predict(image_path, known_embeddings, threshold=0.6):
    image = preprocess_image(image_path)
    unknown_embedding = model.predict(image)

    min_distance = float('inf')
    identity = "unknown"

    for name, embedding in known_embeddings.items():
        distance = euclidean(embedding, unknown_embedding)
        if distance < min_distance:
            min_distance = distance
            identity = name

    if min_distance < threshold:
        return identity
    else:
        return "unknown"


# 示例用法
# 加载已知人脸数据
known_faces = [['saitama', 'saitama.jpg']]
known_embeddings = load_known_faces(known_faces)

result = predict('saitama.jpg', known_embeddings)
print("预测结果：", result)
