# Workshop: Image Data Augmentation

Welcome to the Image Data Augmentation workshop! Feel free to work ahead following the directions in this readme document.

## Prerequisites

Please create accounts (if you do not already have them) on the following sites:

* [Edge Impulse](https://www.edgeimpulse.com/) (to train machine learning models)
* [Google Gmail](https://gmail.com/) (for use with Google Colab)

## 1: Examine Old Model

Navigate to the following project: [https://studio.edgeimpulse.com/public/36514/latest](https://studio.edgeimpulse.com/public/36514/latest)

Take a look at the training and test datasets in *Data acquisition*. How many samples are there per class? Do you think this is enough to create a robust model?

Go to *Impulse design > NN Classifier*. Take a look at the model: it is the default convolutional neural network (CNN) used by Edge Impulse for image classification. What does the confusion matrix (CM) tell us about the model accuracy? Do you think there is overfitting?

![Edge Impulse trained model](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-01.png)

Go to *Model testing*. Click **Classify all** to perform inference on the test set samples. Do you see any issues with the test set performance?

![Edge Impulse trained model](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-02.png)

## 2: Look at Saliency Map and Class Activation Map (CAM)

Find one of the test samples that was mis-classified (e.g. 19.png). Click the three vertical dots and click **Show classification**. A new tab should open that will perform inference on just that sample.

Any thoughts as to why that sample might have been misclassified?

Right-click on the image and select **Save image as...**. Save the image somewhere on your computer (e.g. Downloads folder).

![Edge Impulse trained model](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-03.png)

Go to *Dashboard*. Download the **TensorFlow SavedModel**.

![Edge Impulse trained model](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-04.png)

Open the [Saliency and Grad-CAM Examples Colab notebook](https://colab.research.google.com/github/ShawnHymel/ei-workshop-image-data-augmentation/blob/master/workshop_01_saliency_and_grad_cam.ipynb). 

Click the folder icon on the left side to open the Files browser. Click the upload icon to open a browser window. Select the zipped model file and image (e.g. ei-electronic-components-cnn-nn-classifier-tensorflow-savedmodel-model.zip and 19.png). These should appear in the Files browser.

![Edge Impulse trained model](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-05.png)

Note that a Colab notebook runs in a Linux environment. The files that you just uploaded can be found in the */content/* directory.

Click on the first cell with code. Press *shift + enter* to run the cell (or you can click the play icon in the upper-left corner of the cell). If you get a pop-up saying that the notebook was not authored by Google, click *Run anyway*.

In the next cell, make sure that the settings match the project information. As long as you did not rename anything, the filenames should be the same.

You might need to change the `IMAGE_PATH` variable to point to the test sample image that you just uploaded. Similarly, change the `TRUE_LABEL` variable to the ground truth label of the image.

![Edge Impulse trained model](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-06.png)

Run the rest of the cells in order, and pay attention to the output. The first section should show you the original image with a saliency map overlay.

**Saliency Map**: which pixels in the input image were most important. See [this article](http://www.scholarpedia.org/article/Saliency_map) to learn more about saliency maps.

The second section should show you the original image with a class activation map overlay (specifically using the Grad-CAM algorithm).

**Class Activiation Map**: with features from the final convolution step were most important. See [this article](https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/) to learn more about Grad-CAM.

![Edge Impulse trained model](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-07.png)

What are these maps telling us about where the model is looking in each image to make its classification decision?

## 3: Examine Augmentation Techniques



## License

Unless otherwise noted, all code in this repository is licensed as follows:

Copyright 2021 EdgeImpulse, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.