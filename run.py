# uses https://github.com/ido-ran/google-photos-api-python-quickstart/blob/master/quickstart.py
# https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/

#import libraries
import numpy as np
from MyMLP import MLP, Normalization, process_label
from visualization import plot2d, plot3d
from PIL import Image
from matplotlib import image, pyplot
import os

# import images and convert to numpy arrays, in the future we can have all images in a pool and implement a split function to create test/validate/train
training_images = []
for filename in os.listdir('train'):
    img_data = image.imread('train/' + filename)
    training_images.append(img_data)

test_images = []
for filename in os.listdir('test'):
    img_data = image.imread('test/' + filename)
    test_images.append(img_data)

valid_images = []
for filename in os.listdir('valid'):
    img_data = image.imread('valid/' + filename)
    valid_images.append(img_data)

# generate larger dataset through manipulation of images


# read in data.
# training data
train_data = training_images
train_labels = np.genfromtxt('train/training_labels.txt',delimiter=',')

# validation data
valid_data = np.genfromtxt("optdigits_valid.txt",delimiter=",")
valid_x = valid_data[:,:-1]
valid_y = valid_data[:,-1].astype('int')

# test data
test_data = np.genfromtxt("optdigits_test.txt",delimiter=",")
test_x = test_data[:,:-1]
test_y = test_data[:,-1].astype('int')

# normalize the data
normalizer = Normalization()
normalizer.fit(train_data)
train_data = normalizer.normalize(train_data)
valid_x = normalizer.normalize(valid_x)
test_x = normalizer.normalize(test_x)

# process training labels into one-hot vectors
train_labels = process_label(train_labels)

############### Problem a ###################
# experiment with different numbers of hidden units
candidate_num_hid = [4,16,20,24,32,48]
valid_accuracy = []
for i, num_hid in enumerate(candidate_num_hid):
    # initialize the model
    clf = MLP(num_hid=num_hid)
    # update the model based on training data, and record the best validation accuracy
    cur_valid_accuracy = clf.fit(train_data,train_labels,valid_x,valid_y)
    valid_accuracy.append(cur_valid_accuracy)
    print('Validation accuracy for %d hidden units is %.3f' %(candidate_num_hid[i],cur_valid_accuracy))

# select the best number of hidden unit and use it to train the model
best_num_hid = candidate_num_hid[np.argmax(valid_accuracy)]
clf = MLP(num_hid=best_num_hid)
_ = clf.fit(train_x,train_y,valid_x,valid_y)

# evaluate on test data
predictions = clf.predict(test_x)
accuracy = np.count_nonzero(predictions.reshape(-1)==test_y.reshape(-1))/len(test_x)

print('Test accuracy with %d hidden units is %.3f' %(best_num_hid,accuracy))


############### Problem b ###################
# visualization for 2 hidden units
clf = MLP(num_hid=2)
_ = clf.fit(train_x,train_y,valid_x,valid_y)
# validation set visualization
hid_features = clf.get_hidden(valid_x)
plot2d(hid_features,valid_y,'valid')


# visualization for 3 hidden units
clf = MLP(num_hid=3)
_ = clf.fit(train_x,train_y,valid_x,valid_y)
# validation set visualization
hid_features = clf.get_hidden(valid_x)
plot3d(hid_features,valid_y,'valid')
