from ultralytics import YOLO, checks
import torch

checks()

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load a model
model = YOLO('./yolov8s.pt')  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data='gym_equipment/data.yaml', epochs=5)  # train the model
print(results)
# results = model.val(data='gym_equipment/data.yaml')  # evaluate model performance on the validation set
# print(results)
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
print(results)
# results = model.export()  # export the model to ONNX format
model.save("gym.pt")