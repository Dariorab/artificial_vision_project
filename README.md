# Artificial Vision Project
## Introduction


## Authors and Acknowledgment
Project contributors:
- **[Mariniello Alessandro](https://github.com/alexmariniello)**
- **[Pepe Paolo](https://github.com/paolopepe00)**
- **[Rabasca Dario](https://github.com/Dariorab)**
- **[Vicidomini Luigi](https://github.com/luigivicidomini)**

Thank you to all the contributors for their hard work and dedication to the project.

# How To Execute
To execute the project, you need to install the libraries listed in the `requirements.txt` file. 
You can do this by running the following command:

```bash
pip install -r requirements.txt
```

Download the weights of the PAR model from the following link:
[Download Weights](https://drive.google.com/drive/u/1/folders/1-TtZmkiNiRreebRfynGHOwAR9WUIlt-I)

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
python training.py 
```

Notes:
* Change dataset path with yours.

## Test


## Content

