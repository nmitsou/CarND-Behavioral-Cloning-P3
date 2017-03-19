# Behavioral cloning

The current project provides a solution to the project 3 / Term 1 of the Self-driving car nano-degree by udacity.
The goal of the project is to train a deep neural network to learn to predict the driving angle given an image from the car's camera.

## Deep Network Model

I used the Comma.ai model since it has been proven to work well on a similar problem. The only modification was the input image size.

In short the comma.ai deep learning model consists of a sequence of three convolution layers and ELUs. The output of this network is flattened and passed into a neural network with one hidden layer and drop-out and ELU layers between the input hidden and hidden output layers.

The architecture of the network is shown below:
![png](images/model.png)


## Data collection

To train the network, I used the dataset provided by udacity. I collected additional data mostly on cases where the car failed to drive autonomously (turns). 

## Data evaluation

By examining the udacity dataset, I notice that the dataset is unbalanced. There are a lot of cases of zero angles and ... left turns...

## Data pre-processing

To balance the dataset, I filter the original dataset in order to randomly remove zero.. Moreover, to balance the left and right turns, I vertically revert the images and add them with the opposite sign to the dataset. 
