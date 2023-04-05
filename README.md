# AMLS II Assignment 2022/23

## Cassava leaf disease classification
This is my approach to solving the problem of cassava leaf disease classification posted on [Kaggle](https://www.kaggle.com/competitions/cassava-leaf-disease-classification). I have approached the problem with a pretrained EfficientNetB0 model and then with a custom convolutional neural network without any pretrained weights.

### Organization of the project

[main.ipynb](main.ipynb) - This is the main functional file in a Jupyter Notebook format. It contains data exploration as well as model training and result presentation.

[train.csv](train.csv) - This is a spreadsheet file which contains data description including the list of images used for training and their corresponding labels.

[label_num_to_disease_map.json](label_num_to_disease_map.json) - This file is a mapping between class labels and cassava disease names.

[results](results) - In this folder there are two subfolders, one containing [saved models](results/models/) and the other one [generated plots](results/graphs/).

### Python packages required 
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [TensorFlow and Keras](https://www.tensorflow.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)