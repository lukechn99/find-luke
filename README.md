# Guide

To use, add your own images into the train, test, and valid folders. Name images with the label followed by an underscore and id number. Something like "luke_01.jpg" would work. This needs to be done for all data except for test data.

## Structure

Following the conventional method of ML, we follow the steps of...

1. Prep data (import, max pool, normalize)

2. Fit the model

3. Import test images

4. Perform edge detection using a Prewitt or Sobel kernel to select areas of interest (these areas will be weighted by prominence when classifying the image)

5. Use the model to classify each area/image
