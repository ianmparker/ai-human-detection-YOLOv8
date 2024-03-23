# ai-human-detection-YOLOv8

My LinkedIn : https://www.linkedin.com/in/ian-parker-596011142/

---------------------------------------
# Overview

This repository contains a YOLO model trained on a dataset of human images.

The trained Model is then used to detetct and annotate humans in each frame along with the Model's confidence that each of those objects is a human. 

Each human detected is then counted and the total is displayed the console.

I have attatched examples of the model at work below. 

  - The first of which was from the 2024 NESBE Convention (unfortunately the video was too big for github)
  - The second uses footage from a busy Times Square in New York. 
---------------------------------------
# Setup/Installation

You may need to do some of the folowing in order to get this code to run, depending on how you currently have your enviorment set up. 

  - Use a virtual enviorment
  - Install python 3.7.0 or later
  - Install PyTorch 1.7 or later
  - Install Darknet/ultralytics

You can also use your own video by changing the video path

![human detection video import](https://github.com/ianmparker/human-detection-YOLOv8/assets/18231849/9cc7769a-75c5-4bdc-92f0-6361963098a4)


---------------------------------------
# Screenshots

![NSBE 2024 Convention Ford Booth](https://github.com/ianmparker/ai-human-detection-YOLOv8/assets/18231849/8ad1e7d6-419c-4e92-858d-60144e01b536)

2024 NSBE Convention Ford Booth 

![Times Square Crowd](https://github.com/ianmparker/human-detection-YOLOv8/assets/18231849/371521fb-801f-4dad-a0e9-b6f20d220685)

Times Square Crowd

------------------------------------

# References: 

  - Koby_n_Code Computer Vision Tutorial : https://www.youtube.com/watch?v=hg4oVgNq7Do
  - Official Yolo Github: https://github.com/ultralytics/ultralytics
  - Oficcial Darknet YOLO Website : https://pjreddie.com/darknet/yolo/
  - Times Square Crowd - https://www.youtube.com/watch?v=P0wNIsAjht8
