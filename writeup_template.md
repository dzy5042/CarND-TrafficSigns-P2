#**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/exp1.png "OriginalPlot"
[image2]: ./examples/exp2.png "GrayscalingPlot"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./german traffic signs/30.jpg "Traffic Sign 1"
[image5]: ./german traffic signs/construct.jpg "Traffic Sign 2"
[image6]: ./german traffic signs/kindergarten.jpg "Traffic Sign 3"
[image7]: ./german traffic signs/STOP.jpg "Traffic Sign 4"
[image8]: ./german traffic signs/X.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The number of training examples is 34799.
* The number of validation examples is 4410.
* The number of testing examples is 12630.
* The Image data shape is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### Data Visualization

Here is one traffic sign data in training set.

![120 speed limit sign][image1]

###Design and Test a Model Architecture

####Data preprocess

I decided to normalize and grayscale the images because because this will eliminate some misleading information like color of the sign.

Here is an example of a traffic sign image before and after grayscaling.
![before][image1]
![after][image2]


####Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 1 channel grayscale image   			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU with dropout		| 0.7 dropout									|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU with dropout		| 0.7 dropout  									|
| Max pooling			| 2x2 stride,  outputs 5x5x16					|
| Flattern				| outputs 400									|
| Fully Connected		| outputs 120									|
| RELU with dropout		| 0.7 dropout									|
| Fully Connected		| outputs 84									|
| RELU with dropout		| 0.7 dropout									|
| Fully Connected		| outputs 43									|
 


To train the model, I used an optimizer with loss based on reduced mean of cross entropy. Batch size is 128, learning rate is 0.001, the model has been trained with 50 epochs.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.958
* test set accuracy of 0.948

I start with the LeNet model as the starting point. I found 10 epoch is not enough for model to get fully trained. I increased epoch to 50 and then I found I can achieve training set accuracy of 0.994 and validation accuracy less than 0.9, which is an indicator that my model suffer from overfit. Then I choosed dropout method to regularize my model. I tried both 0.5 and 0.7 dropout rate and they both works well. 0.7 converges a little faster.
 

###Test a Model on New Images


Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]



Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 Speed Limit   		| 30 Speed Limit								| 
| Road Work    			| Road Work										|
| Children Crossing		| Children Crossing								|
| Stop		      		| Stop							 				|
| Yield					| Yield			      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%!

####Softmax Probability Analysis
The code for making predictions on my final model is located in the 39th cell of the Ipython notebook.

For all images model give the highest probability larger than 0.99, which means model is very confident on the prediction. This may due to the images are in well condition for recolonization.
