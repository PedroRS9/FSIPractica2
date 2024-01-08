# FSI Practica 2

This is a repository for the second practice of the subject "Fundamentos de los Sistemas Inteligentes" in the University of Las Palmas de Gran Canaria.
The practice consists in making neuronal networks for classifying images of a vegetables dataset downloaded from Kaggle: [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset). The dataset is composed of 15 vegetables: Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish and Tomato.

## Getting Started

### Prerequisites
Ensure you have Python 3.9+ installed on your system. You can download Python [here](https://www.python.org/downloads/).

### How to install

First of all, you need to clone the repository.

```
git clone https://github.com/PedroRS9/FSIPractica2.git
```

Then, you need to install the requirements:

```
pip install -r requirements.txt
```

### How to run

```commandline
python main.py
```

### Explanation of each file

#### config.py

This file contains the hyperparameters of the neuronal network. You can change the hyperparameters in this file.

#### data_loader.py
This file contains the code for loading the dataset. It also contains the code for splitting the dataset into train, validation and test.

#### main.py
The main entry point of the project. Orchestrates the training process, evaluates the model's performance, and triggers the plotting of results.

#### model_builder.py
This file contains the code for creating the keras model of the convolutional neuronal network.

#### plotting.py
This file contains the code for plotting graphs (training history and confusion matrix).

### Authors
* **Pedro Romero Suárez** - [PedroRS9](https://github.com/PedroRS9)
* **Nahuel Cosme Díaz Vera** - [NahuelCosmeDiazVera](https://github.com/NahuelCosme-DiazVera)
