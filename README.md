# Behavioral cloning

The current project provides a solution to the project 3 / Term 1 of the Self-driving car nano-degree by udacity.
The goal of the project is to train a deep neural network to learn to predict the driving angle given an image from the car's camera.

## Deep Network Model

I used the Comma.ai model since it has been successfully used on a similar problem. The only modification was the input image size.

In short the comma.ai deep learning model consists of a sequence of three convolution layers and ELUs. The output of this network is flattened and passed into a neural network with one hidden layer and drop-out and ELU layers between the input hidden and hidden output layers.

The architecture of the network is shown below:

![png](images/model.png)


## Data collection

To train the network, I used the dataset provided by udacity. I used the center images together with the left and right camera image with added value on the angle +/- 0.25 rad. I collected additional data mostly on cases where the car failed to drive well autonomously (e.g. on turns). 


## Data evaluation

By examining the udacity dataset, I notice that the dataset is unbalanced. There are a lot of cases of zero angles (shown below).

![png](images/histogram_original.png)

## Data pre-processing

To balance the dataset, I filter the original dataset in order to randomly remove zero angle instances. 
The histogram of the filtered dataset is shown below: 

![png](images/histogram_filtered.png)

Moreover, to balance the left and right turns, I vertically revert the images and add them with the opposite sign to the dataset. 
The histogram of the final dataset is shown below: 

![png](images/histogram_left_right.png)

### Image pre-processing

The image from the camera of the car is pre-processed in the following way:

- crop the image ignoring the upper part (sky) and the lower part (car chasis) of the image
- downsample the image by skipping every second column

| Input image | Cropped image |
|---|---|
| ![png](images/example_input.png) | ![png](images/example_cropped.png) | 


