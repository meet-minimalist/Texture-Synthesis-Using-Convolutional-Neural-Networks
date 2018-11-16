# Texture Synthesis Using Convolutional Neural Networks
Tensorflow implementation of paper - "Texture Synthesis Using Convolutional Neural Networks"


#### In this notebook, we'll generate new textures based on the given texture. The output will be generated from the scratch noisy image. The steps of the process is as follows. Also, the notebook is created to facilitate self-learning approach. 

_Step 1: Preprocessing the input image_

_Step 2: Computing the output for all the layers for the input image._

_Step 3: What is loss function in this problem and computing the loss function._

_Step 4: Running Tensorflow model to minimize the loss and optimize the input noise variable._

_Step 5: Post processing and displaying the image._

_Step 6: Automating the stuffs_

_Step 7: Plotting the successful results._

##### Results:

![](https://github.com/meet-minimalist/Texture-Synthesis-Using-Convolutional-Neural-Networks/blob/master/compiled%20results%20-%201.png)

##### Files:
1. helper.py - Used for pre-processing the image and post-processing the image
2. tf_helper.py - Used to compute the layer wise output for a given texture sample image
3. paper folder - contains the paper and its important cutouts
4. tensorflow_vgg folder - contains the helper vgg16_avg_pool.py function to load the pre-trained weights ".npy" file
5. image_resources/original - contrains original images as well as cropped in aspect ratio of 1:1 to use them 
6. image_resources/processed - contains down-sized images to be feed in to the CNN model
7. image_resources/outputs - contains synthesised texture outputs for different texture input images 


##### References:
1. Paper link: https://arxiv.org/abs/1505.07376

2. VGG16 Tensorflow Model - https://github.com/machrisaa/tensorflow-vgg  
  Pre-trained VGG16 tensorflow model along with helper files. Big shoutout to the owner. Also, vgg16.npy can be downloaded from the link provided in this repository. I have modified the vgg16.py file to facilitate average pooling instead of max pooling.
