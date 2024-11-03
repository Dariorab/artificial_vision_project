# Artificial Vision Project
## Introduction
This project aims to develop an artificial vision system for tracking individuals in a scene 
and analyzing their behaviors. The system will assign unique identifiers to each person 
for consistent tracking over time. It will monitor interactions within specified Regions of Interest (ROIs), 
as defined in a JSON configuration file, and assess how often individuals enter these areas 
and the duration of their presence.

Additionally, the system will classify key attributes for each person, including gender, 
the presence of a bag or backpack, the presence of a hat, and the colors of their upper and lower clothing. 
This approach provides valuable insights into individual behaviors and characteristics, 
making the system useful for applications in surveillance, retail analytics, and public safety.

_**More details about choosen made can be found in the `report.pdf` file.**_

## Authors and Acknowledgment
Project contributors:
- **[Mariniello Alessandro](https://github.com/alexmariniello)**
- **[Pepe Paolo](https://github.com/paolopepe00)**
- **[Rabasca Dario](https://github.com/Dariorab/index_repositories)**
- **[Vicidomini Luigi](https://github.com/luigivicidomini)**

Thank you to all the contributors for their hard work and dedication to the project.

# How To Execute
To execute the project, you need to install the libraries listed in the `requirements.txt` file. 
You can do this by running the following command:

```bash
pip install -r requirements.txt
```

Download the weights of the PAR model from the following link:
[Download Weights](https://drive.google.com/file/d/1AjH-RlhfxMVpTJfsasAFuB0Ww8DYvgA0/view?usp=drive_link)

`yolov8m` and `osnet` weights can be dowloaded from their official site.

Then, after previous step,
you can run the project by executing the following command:

```bash
python main.py --video video/video.mp4 --configuration configuration/configuration.txt
--results results/results.txt
```

### Notes:
* `video` folder you must put the video to analyze.
* in the `configuration` folder you must put the configuration file of the regions of interest.
* in the `results` folder you must insert the name of the file of output to create after the execution of the program.



## Training
If you want to train the model, you can run the following command:

```bash
python training.py --epochs 50 --batch_size 128 --num_workers 2
```

**Arguments:**
* `--lr` learning rate of the model.
* `--lr_classificator` learning rate of the classificator.
* `--min_loss` minimum loss to save the model.
* `--epochs` number of epochs to train the model.
* `--batch_size` batch size of the training.
* `--result` path to save the results.
* `--early_stopping` number of epochs to stop the training if the loss does not decrease.
* `--num_workers` number of workers to load the data.
* `--drive` if you run training on google colab.
* `--checkpoint` if you want to load a checkpoint to continue the training.


Notes:
* Change dataset path with yours.
* `annotations` folder contains the annotations of the dataset.
* Download data from the following link:
[Download Data](https://drive.google.com/file/d/1JPIhm5zvWSwYjAdr7rKhrNDF1D9fefPa/view?usp=sharing)


## Test
If you want to test the model, you can run the following command:

```bash
python test.py
```

Notes:
* Download data for testing from the following link:
[Download Data](https://drive.google.com/file/d/1JPIhm5zvWSwYjAdr7rKhrNDF1D9fefPa/view?usp=drive_link)
* annotations are already in the folder `annotations`.

**Arguments**:
* `--batch_size` batch size of the testing.

# Content
The project is divided into the following parts:

## Multi-Task network
The network is used for **PAR** (Pedestrian Attribute Recognition). 
It is composed of two files:
* `classification_head.py` contains the class for implementing single head.
* `multi_task_network.py` contains the class for implementing the multi-task network
which is composed of 5 heads.

## Object Detection and Re-Identification
The object detection and re-identification tasks are handled in the `main.py` file, 
utilizing the YOLO model for object detection and the OSNet architecture 
for person re-identification.

## Dataset
The first step regards to creation of correct annotations
for the dataset and remove those images that don't contain all the attributes.
After that, the dataloaders are created using the `Dataset` class 
in the `mivia_dataset.py` file.

## Training
In the `training.py` file, you will find the complete code structure 
for training a multi-task neural network using GradNorm as the loss function

## Testing
In the `test.py` file, you can find the code for testing PAR model.







