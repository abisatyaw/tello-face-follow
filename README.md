
# Tello Face Follower
Autonomous control of DJI Tello Based on Face Recognition.
#### Goals  
DJI Tello will try to keep the recognized face in the center of the frame




## Acknowledgements
Build with :
 - [opencv](https://github.com/opencv/opencv)
 - [djitellopy](https://github.com/damiafuentes/DJITelloPy)


## Installation

Install this project using [Rye](https://rye.astral.sh/guide/installation/) python package manager if you have by cloning then syncing the project, otherwise install the dependency from requirements.txt

Clone the project

```bash
git clone https://link-to-project
```

Go to the project directory

```bash
cd my-project
```

Install dependencies :

using Rye
```bash
rye sync
```
using pip
```bash
pip install requirements.txt
```  

## Getting Started
first of all we need to train a model to recognize a face
* run capture_dataset.py (make sure you have your webcam enabled)
```bash
rye run python capture_dataset.py
```
train the dataset
* run face_trainer.py
```bash
rye run python face_trainer.py
```
(Optionally) run a test to logic using webcam
* run webcam_test.py
```bash
cd webcam_test
```
```bash
rye run python webcam_test.py
```

run main application and connect Tello to PC to see it live in action
* run main.py
```bash
rye run python main.py
```


## Demo

![alt text](Drone_Test_Fly.gif)
