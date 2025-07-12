# Deep Learning-Driven Virtual Fashion Fitting

## Project Goal

Develop web-based platform where users can virtually try-on clothes in real time, users has to provide two necessary thinks to the system, image and a description, then the system will use computer vision to detect the body structure and use generative ai to generate personalized clothing that fits the user's image

the project will leverage a combination of <strong>Computer Vision</strong> (body segmentation, pose estimation, and clothing detection) and <strong> Gererative AI</strong> (text to image models)
<br>

- <strong>Generative Ai</strong> Generate clothing images based on the user description (e.g., red hat and white t-shirt)

- <strong>Computer Vision</strong> use the user's upload image to detect their body and ensure that the clothing generated fits naturally in term of shape and pose

## Project Development

The project divided into 7 main sections:

### Step1: Data collection

we need two type of datasets, ***Clothing Dataset*** and ***Pose and Segmentation Dataset***

- **Clothing Dataset** large dataset of clothing image and descriptions
    - **DeepFashion Dataset** diverse categories of fashion images with annotations.
    - **Fashion-Gen Dataset** consists of text-image pairs from fashion domain

- **Pose and Segmentation Dataset** datasets for human body pose detection and segmentation to recognize body parts
    - **COCO dataset** large scale dataset for object detection, segmentation and pose estimation 
    - **LIP dataset** dataset for human parsing, containing annotated images for pixel-wise human segmentation.

### Step2: Data Preprocessing and labeling 

- **Image preprocessing** resize all the images into a unique format (e.g., 512x512) and clean up annotation

- **Text preprocessing** normalize and clean user text descriptions so they can be matched to known clothing categories in our system. (e.g., "red shirt", "floral dress", "jeans with ripped knees").

- **Pose Estimation & Segmentation Preprocessing**

    - Use **OpenPose** or **MediaPipe** To detect where body parts are (head, shoulders, elbows, etc.) using an image.

    - Use **semantic segmentation** (e.g., DeepLab) To separate the body parts (e.g., upper body, arms, legs) from the background and from each other in the image.