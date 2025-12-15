# Overview
This program trains a neural network to approximate the output value of a function

# Dependencies
This program requires Python to run, which can be downloaded for free from https://www.python.org/
It also requires the package numpy, which can be installed by running the following line in your command terminal:
pip install numpy

# Operation and Customization
To train the neural network, run main.py
To generate new training data, run createTestData.py
To modify how the model will be trained, edit any or all of the values listed in config.py

## Saving and Loading
The program will load training data from the file specified in TRAINING_DATA_FILE
The program will treat the column from that file labeled with the name specified in TRAINING_DATA_LABEL as labels, and all other columns as features
The program will load the activation functions, weights, and biases from the file specified in NEURAL_NETWORK_FILE
If PRETRAINED_NEURAL_NETWORK is set to false, it will generate new weights and biases; otherwise, it will use the ones contained in the file
After the program is done with training, it will save the neural network it has created to the file specified in SAVE_TO_FILE


## Neural Network Training
The program will train the model according to a loss function listed in LOSS_FUNCTION. The loss functions currently available are L1LossFunction, L2LossFunction, and LogLoss
The model will be trained with a learning rate listed in LEARNING_RATE
The program will try to train the neural network until its loss on a validation set is less than or equal to TRAINING_COMPLETE_THRESHOLD
If this threshold is not met, it will halt training after MAX_EPOCHS have elapsed

## Data Generation
Running the file createTestData.py will fill the file listed in TRAINING_DATA_FILE with new data that the model will be trained on.
This model will have 3 values x, y, and z between 0 and e^TRAINING_DATA_MAX_EXPONENT
The label value will then have the label value calculated according to the formula f(x,y,z)=(x-1)/1+(x-2)/2+(x-3)/3 where x-1, y-2, and z-3 are all clamped to a minimum value of 0
Each of these 4 variables (x, y, z, and label) will then be multiplied by a value between 1-TRAINING_DATA_MULTIPLICATIVE_VARIANCE and 1+TRAINING_DATA_MULTIPLICATIVE_VARIANCE
These data points will also have a value added to them between -TRAINING_DATA_ADDITIVE_VARIANCE and TRAINING_DATA_ADDITIVE_VARIANCE