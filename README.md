
# RT-DETR and DeepSORT: Real-Time Detection Transformer and Tracking

Nowadays, object detection plays a crucial role in enabling computers to understand the visual world. It achieves this by identifying and locating objects within images or videos. However, simply identifying objects in a single frame isn't enough. Object tracking allows us to see how objects move and interact with their surroundings. This is essential for tasks like self-driving cars and video surveillance.

In our project, we leverage a vision transformer-based real-time object detector called REDETR to identify objects. Vision transformers have achieved state-of-the-art (SOTA) performance in object detection. To track the detected objects, we employ the DeepSORT algorithm. DeepSORT excels at assigning unique identifiers to each object, allowing it to differentiate between them even in crowded scenes.

![rtdetr](https://github.com/codershreya/Object-Tracking-Using-ViT-and-Deepsort/assets/93388678/6373364e-38e2-48fa-9fcd-16ce2d130357)
<div align="center">RT-DETR</div>

<br/>

![deepsort](https://github.com/codershreya/Object-Tracking-Using-ViT-and-Deepsort/assets/93388678/93f795cd-98f8-452a-959b-89bbb1423550)
<div align="center">DeepSORT</div>


## Resources

This project utilizes the following resources:

* **RT-DETR Large `rtdetr-l.pt`:** This model is included in the Ultralytics Python API.
  * A list of available pretrained RTDETR models can be found on the Ultralytics [website](https://docs.ultralytics.com/models/rtdetr/#pre-trained-models).
* **Feature Extraction Model:** You can find the CNN checkpoint file [here](https://drive.google.com/drive/folders/1m2ebLHB2JThZC8vWGDYEKGsevLssSkjo).
* The original repository of DeepSORT can be found [here](https://github.com/nwojke/deep_sort).

## Installation

Prerequisites:
- Python 3.8 or later (Make sure you have python installed. Check by running `python --version`)

Steps:
1. Clone the Repository:
```
git clone https://github.com/codershreya/Object-Tracking-Using-ViT-and-Deepsort.git
```

2. Install PyTorch
<br/> - Follow the instructions on the official PyTorch website to install the appropriate version: [Link](https://pytorch.org/get-started/locally/)
<br/> - **Note:** If you don't have an NVIDIA GPU, PyTorch will automatically use your CPU for computations.

4. Install project dependencies:
```
pip install -r requirements.txt
```

4. Verify `deep_sort` folder:
Make sure the `deep_sort` folder is present in the same directory as your main python script (`main.py`)

5. Provide video path:
In `main.py`, replace `'YOUR_PATH_GOES_HERE'` with the actual path to the video
```python
transformer_detector = DETRClass('YOUR_PATH_GOES_HERE')
```

6. Run the project:
```python
python main.py
```
A dialog box will appear after the program finishes loading.

**Note:** Processing with the CPU might take slightly longer compared to using an NVIDIA GPU.

## Demo
![image](https://github.com/codershreya/Object-Tracking-Using-ViT-and-Deepsort/assets/93388678/57a4cb83-1123-4eaa-81e7-838e595e57de)
<div align="center">Real Time Object Detection and Tracking</div>

