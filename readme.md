## Main Code
* [train gym equipment model](train_gym_equipment.py)
* [timing and pose and equipment detect](pose_detect.py) 
* ~~[import_faces_from_video](import_faces_from_video.py)~~ 
* ~~[import_faces_image](import_faces_image.py)~~ 
* ~~[face_recognition_from_video](detect_from_video.py)~~ 
* [Dependencies of the project](requirements.txt) 

### Main Files
* [Model for fitness equipment detection](best.pt)
* [yolov8 model](yolov8n.pt)

## Division of tasks
### Research on technical programmes

**Members**: Jin, Chongyi, Jianlai, Harshini

**Task Description**. 
During the technical solution research phase, our team conducted in-depth research and evaluation of a variety of technologies. The following are the main technologies we researched:

1. **YOLO (You Only Look Once)**.
    - **Overview**: YOLO is a real-time target detection system known for its efficiency and accuracy. We focus on the YOLOv8 version, which has significant improvements in accuracy and speed.
    - **Application**: We use YOLO for gym equipment detection to ensure that the model is able to quickly recognise a variety of fitness equipment.

2. **Face Recognition**.
    - **Overview**: Face Recognition is a dlib-based Python library that enables efficient face detection and recognition. Although we implemented the face recognition functionality, we did not end up integrating it into the project based on our instructor's advice.
    - **Application**: Face recognition was initially considered for user authentication and personalised fitness plan recommendation, but was not ultimately adopted due to project requirement adjustments.

3. **MediaPipe**.
    - **Overview**: MediaPipe is a cross-platform open source framework introduced by Google for building multimodal ML pipelines. We mainly use its Pose module for human keypoint detection.
    - **Application**: We use MediaPipe for human motion detection, such as dumbbell curl and leg press, to recognise and count movements by detecting changes in keypoints.

4. **OpenCV (cv2)**.
    - **Overview**: OpenCV is an open source computer vision and machine learning software library that provides a rich set of image processing and computer vision capabilities.
    - **Application**: We use OpenCV for image preprocessing and enhancement, as well as assisting MediaPipe for keypoint detection and motion recognition.

### Dataset Preparation

**Members**: Jin, Harshini

**Task Description**.
In the dataset preparation phase, we constructed a high-quality training dataset by combining the open source dataset and the images we took at the gym. The specific steps are as follows:

1. **Open source dataset**.
    - **Source**: We collected image datasets of gym equipment and actions from multiple open source platforms, such as Kaggle, Google Dataset Search, etc.
    - **DATA PROCESSING**: We clean and organise the open source data to ensure uniform data format and remove blurred or duplicate images.

2. **Self-acquired data**.
    - **Shooting**: We took a large number of images in actual gym environments, covering a variety of fitness equipment and user actions. To ensure the diversity of the data, we shot at different time periods and angles to try to cover all possible usage scenarios.
    - **Added to test set**: We add self-collected data to the test set to ensure that the performance of the model in real scenarios can be effectively verified.

3. **Data integration**.
    - **Merge**: Integrate the open source data with the self-picked data to build a comprehensive dataset with rich samples.
    - **Division**: Divide the dataset into training, validation, and testing sets to ensure the scientific accuracy of model training and evaluation.

### Model Training

**MEMBERS ONLY**: Jin, Jianlai

**Task Description**.
- Select and build appropriate model architecture and implement model training.
- Monitor the training process and adjust hyperparameters and model structure to improve model performance.
- Evaluate the training effect, save and manage the trained models.


### Write major device detection and motion detection and timing code

**Members**: Jin, Jianlai, Chongyi

**Task Description**.
- Write and debug the model and related code to solve various technical problems that arise.
- Optimise code efficiency and performance (code will be tested every three frames using the model) to ensure the model runs smoothly in the demo application.
- Test and verify the optimisation effect to ensure the accuracy and reliability of the model.

### Record Demo Video

**Members**: Jin, Jianlai

**Task Description**.
- Record a demo video showing the project results, including model effects and application scenarios.