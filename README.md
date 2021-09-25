# Workshop: Image Data Augmentation

Welcome to the Image Data Augmentation workshop! Feel free to work ahead following the directions in this README.

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

![Testing trained model in Edge Impulse](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-02.png)

## 2: Look at Saliency Map and Class Activation Map (CAM)

Find one of the test samples that was mis-classified (e.g. 19.png). Click the three vertical dots and click **Show classification**. A new tab should open that will perform inference on just that sample.

Any thoughts as to why that sample might have been misclassified?

Right-click on the image and select **Save image as...**. Save the image somewhere on your computer (e.g. Downloads folder). I recommend renaming the file to something simple (e.g. "19.png").

![Save test sample](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-03.png)

Go to *Dashboard*. Download the **TensorFlow SavedModel**.

![Inference on test sample](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-04.png)

Open the [Saliency and Grad-CAM Examples Colab notebook](https://colab.research.google.com/github/ShawnHymel/ei-workshop-image-data-augmentation/blob/master/workshop_01_saliency_and_grad_cam.ipynb). 

Click the folder icon on the left side to open the Files browser. Click the upload icon to open a browser window. Select the zipped model file and image (e.g. ei-electronic-components-cnn-nn-classifier-tensorflow-savedmodel-model.zip and 19.png). These should appear in the Files browser.

![Download saved model from Edge Impulse](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-05.png)

Note that a Colab notebook runs in a Linux environment. The files that you just uploaded can be found in the */content/* directory.

Click on the first cell with code. Press *shift + enter* to run the cell (or you can click the play icon in the upper-left corner of the cell). If you get a pop-up saying that the notebook was not authored by Google, click *Run anyway*.

In the next cell, make sure that the settings match the project information. As long as you did not rename anything, the filenames should be the same.

You might need to change the `IMAGE_PATH` variable to point to the test sample image that you just uploaded. Similarly, change the `TRUE_LABEL` variable to the ground truth label of the image.

![Change settings in Colab notebook](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-06.png)

Run the rest of the cells in order, and pay attention to the output. The first section should show you the original image with a saliency map overlay.

**Saliency Map**: which pixels in the input image were most important. See [this article](http://www.scholarpedia.org/article/Saliency_map) to learn more about saliency maps.

The second section should show you the original image with a class activation map overlay (specifically using the Grad-CAM algorithm).

**Class Activiation Map**: with features from the final convolution step were most important. See [this article](https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/) to learn more about Grad-CAM.

![Grad-CAM heat map](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-07.png)

What are these maps telling us about where the model is looking in each image to make its classification decision?

## 3: Examine Augmentation Techniques

Open the [Image Transforms Demo Colab notebook](https://colab.research.google.com/github/ShawnHymel/ei-workshop-image-data-augmentation/blob/master/workshop_02_transforms_demo.ipynb).

Examine and run every cell to see some of the different image transformations that are possible.

![Image data augmentation with Colab](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-08.png)

See [this article from Nanonets](https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/) if you would like to learn more about image augmentation techniques. [This article from Machine Learning Mastery](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/) goes into some advanced augmentation techniques that we do not cover here.

## 4: Generate Augmented Dataset

Open the [Image Data Augmentation Colab notebook](https://colab.research.google.com/github/ShawnHymel/ei-workshop-image-data-augmentation/blob/master/workshop_03_image_data_augmentation.ipynb).

The *Settings* cell should not need to be modified. It points to this repository to download the same dataset used to train the original model. We are going to create an augmented dataset in this notebook.

Run each cell, paying close attention to the *Transform Functions* section. This section contains a number of functions designed to create copies of an input image that are modified in some way. Each function is documented with examples on how to use it, as you will need to call the function for challenge 2.

***Challenge 1***: Right now, only the `create_flipped(img_array)` function is called, which means we're just augmenting our data with flipped copies of each image. Call the other transformation functions (e.g. `create_rotated()`) to add more augmented images to your dataset. Look for the `# >>>ENTER YOUR CODE HERE<<<` comment in the *Do Transforms* section to see where to call these functions. The line above it, `img_tfs.append(create_flipped(img_array))`, should give you an idea of how to call these functions, as `img_tfs` is the collection of images.

***Challenge 2 (Bonus)***: Implement the `create_noisy()` function (in the *Add Noise* section) that returns a list of image arrays that are copies of the original input image with random noise added. Look at the other transformation functions to see how to accept an image array parameter and return a list of image arrays. You also might find the [skimage.util.random_noise()](https://scikit-image.org/docs/dev/api/skimage.util.html#random-noise) function useful.

***Solutions***: See [this notebook](https://colab.research.google.com/github/ShawnHymel/computer-vision-with-embedded-machine-learning/blob/master/2.3.5%20-%20Project%20-%20Data%20Augmentation/solution_image_data_augmentation.ipynb) for the solutions. I recommend trying the challenges yourself before peeking at the solutions!

## 5: Retrain Model with New (Augmented) Data

After adding the function calls (as detailed in *Challenge 1*) and running all of the cells, you should have a file named *augmented_dataset.zip* in your *Files* browser. Right-click the file and download it.

![Download augmented dataset](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-09.png)

Unzip the file on your computer. Feel free to look through the dataset--it should contain the original images along with augmented copies (flipped, rotated, translated, etc.).

Navigate to [Edge Impulse](https://www.edgeimpulse.com/) and sign in to your account. Create a new project. Go to *Data acquisition*, and click on the **Upload existing data** button.

![Upload existing data to Edge Impulse](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-10.png)

Click **Choose Files** and select all of the images in the *background* folder (located in the augmented dataset you just downloaded). Leave *Automatically split between training and testing* selected, and select **Enter label**. Enter **background** as the label. Click **Begin upload**.

![Upload one category of images to Edge Impulse](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-11.png)

Repeat this process for the other categories. Don't forget to change the label to match the category! You should have 5 classes with the respective labels:

* background
* capacitor
* diode
* led
* resistor

Go to *Impulse design*. Change the *Image data* width to **28** and height to **28**. Add an *Image* block as your processing block and a *Classification (Keras)* block for your learning block.

![Create impulse](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-12.png)

Click **Save Impulse**.

Go to the *Image* page under *Impulse design*. Change the *Color depth* to **Grayscale**. Click **Save parameters**.

![Set features for images](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-13.png)

You should be automatically taken to the *Generate features* page. Click the **Generate features** button and wait while the images are converted to grayscale and resized to 28x28.

![Set features for images](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-14.png)

Navigate to *NN Classifier*. Change the *Number of training cycles* to **100**. Click **Start training**. Wait a few minutes while training completes. Take a look at the output confusion matrix: how well did this new model perform versus the old model (even though the layers should be the exact same)?

![View training results](https://raw.githubusercontent.com/ShawnHymel/ei-workshop-image-data-augmentation/master/Images/screen-15.png)

Go to *Model testing* and click **Classify all**. How well does the model perform on the test set? Why do you think it performed better or worse than the original model?

Feel free to generate saliency maps and class activation maps using an image from this new test set. How is the new model making decisions, and how does that differ from the way the old model made decisions?

## License

This README document is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

---

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