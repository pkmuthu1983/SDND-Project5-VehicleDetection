**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_noncar.png
[image2]: ./output_images/hog_car_rgb.png
[image3]: ./output_images/hog_car_YCrCb.png
[image4]: ./output_images/hot_windows.png
[image5]: ./output_images/heat_maps_boundingboxes.png
[video1]: ./project_video.mp4

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Please refer code cell 2 for function which calculates hog parameters (get_hog_features). Note that the most of code in the notebook (Pipeline.ipynb) is from the quizzes and udacity examples.

Below, is an example of a car and a non-car image. 
![Car and non-car image][image1]

The hog features for the car is shown in RGB color space below. The parameters are orientations = 9, pixels_per_cell = 8, cells_per_block = 2.
![HOG features for car in RGB][image2]

Hog features for the same car image in YCrCb space is shown below. Eventhough there is no noticeable visual difference between hog features in these two color spaces, the test accuracy achieved with YCrCb was much higher than RGB.
![HOG features for car in YCrCb][image3]

The final feature set included spatial features (in YCrCb space)
as well as the histogram features in addition to hog features. The spatial feature size was 32 and number of histograms was 32.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried increasing the orientation bin size to 12 and pixels_per_cell to 16, but the test accuracy seemed to decrease. The parameters mentioned in section 1, seemed to give good bounding boxes for the test images.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using feature vectors that included HOG features for all three channels, spatial features, and histogram features. I did not change any parameter of the LinearSVM. Instead, just played with the parameters that affect the feature set. The test accuracy was close to 99%. Parameters are defined in cell 3 and the training code can be found in cell 4.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I generated several windows of size (128,128) and (96,96) within a certain portion of the image, using sliding window technique. The windows of size 128 by 128 overlapped 80% with each other. Refer code cell 6 (function slide_window). I tried different window sizes 96 by 96, 128 by 96 and overlap of 80%, 50%. Larger window sizes lead to bigger boxes combining two cars and smaller ones caused  single to be split into two bounding boxes. The combination 128 and 96 sized seemed to work fine. slide_window function defined in cell 2 finds the windows, and is used in cell 6.

In the next step, I extracted features for the image within each window and passed it to the classifier. The classifier predicts whether each image is a car or not. search_window function is used by the Process_image2 function defined in cell 6

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some example images from video with classifier predictions. The windows which are predicted as car are drawn in blue. As we can see the classifier correctly identifies windows around cars.

![Predictions from classifier][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Final video is [here](./project_video_withvehicledetection.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used heat maps and thresholding to filter out false positives.
The classifier first detects windows which contain cars. I convert this into a heat map, such that pixels around cars will have postive value and those that are not cars will have zero value. The pipeline then averages heat maps over several consective frames to get better idea of cars. False predictions will have very low intensity after averaging. Finally, I apply a threshold to the averaged heatmap, to find cars with high confidence. This thresholded heatmap is passed to the label() method, which produces a bounding box around clusters of non-zero pixels.

### Heatmaps
 
Here are three consecutive frames and their heatmaps:

![Heat maps, and final bounding boxes after averaging][image5]

The pipeline obtains an average of previous five heatmaps and then applies and threshold, and uses the label() function to find bounding box on the heatmap. The bounding boxes are shown in the last row above.

The heat maps shows that there are false positives in the third frame. However, those are removed through averaging and thresholding.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There were several problems, especially with parameter tuning as the parameter space is huge. There is no clear idea how to tune the HOG parameters. The classifier had test accuracy of 99%, but it still performed poorly on video. It takes several frames to get good bounding boxes around cars. That's because, the classifier gives completely different predictions even when there was slight change in road image between two consecutive frames. Furthermore, for some reason it does not detect cars from the side. rear-view predictions are good. I guess more training it required. Another challenge was the time taken to process each image was quite high. I believe deep-learning techinques must be used for faster clasification, and reduce the sensitivity to parameters.

