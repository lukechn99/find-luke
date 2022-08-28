# uses https://github.com/ido-ran/google-photos-api-python-quickstart/blob/master/quickstart.py
# https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/

#import libraries
import numpy as np
from MLP import MLP, Normalization, process_label
from visualization import plot2d, plot3d
from PIL import Image
from matplotlib import image, pyplot
import os

# import images and convert to numpy arrays, in the future we can have all images in a pool and implement a split function to create test/validate/train
training_images = []
training_names = []
for filename in os.listdir('train'):
    try:
        img_data = image.imread('train/' + filename)
        img_data.resize((400, 400))
        img_data = img_data.flatten('C')
        training_images.append(img_data)
        training_names.append(filename.split('_')[0])
    except Exception as e:
        print(e)

valid_images = []
valid_names = []
for filename in os.listdir('valid'):
    try:
        img_data = image.imread('valid/' + filename)
        img_data.resize((400, 400))
        img_data = img_data.flatten('C')
        valid_images.append(img_data)
        valid_names.append(filename.split('_')[0])
    except Exception as e:
        print(e)

test_images = []
test_names = []
for filename in os.listdir('test'):
    try:
        img_data = image.imread('test/' + filename)
        img_data.resize((400, 400))
        img_data = img_data.flatten('C')
        test_images.append(img_data)
        test_names.append(filename.split('_')[0])
    except Exception as e:
        print(e)

# read in data.
# training data
train_data = np.asarray(training_images, dtype=object).astype('int')
train_labels = np.array(training_names)
print(train_data.shape)

# validation data
valid_data = np.array(valid_images, dtype=object).astype('int')
valid_labels = np.array(valid_names)

# test data
# test_data = np.genfromtxt("optdigits_test.txt",delimiter=",")
# test_x = test_data[:,:-1]
# test_y = test_data[:,-1].astype('int')
test_data = np.array(test_images, dtype=object).astype('int')
test_labels = np.array(test_names)

# normalize the data
normalizer = Normalization()
normalizer.fit(train_data)
train_data = normalizer.normalize(train_data)
valid_data = normalizer.normalize(valid_data)
test_data = normalizer.normalize(test_data)

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
    cur_valid_accuracy = clf.fit(train_data,train_labels,valid_data,valid_labels)
    valid_accuracy.append(cur_valid_accuracy)
    print('Validation accuracy for %d hidden units is %.3f' %(candidate_num_hid[i],cur_valid_accuracy))

# select the best number of hidden unit and use it to train the model
best_num_hid = candidate_num_hid[np.argmax(valid_accuracy)]
clf = MLP(num_hid=best_num_hid)
_ = clf.fit(train_data,train_labels,valid_data,valid_labels)

# evaluate on test data
predictions = clf.predict(test_data)
accuracy = np.count_nonzero(predictions.reshape(-1)==test_labels.reshape(-1))/len(test_data)

print('Test accuracy with %d hidden units is %.3f' %(best_num_hid,accuracy))


############### Problem b ###################
# visualization for 2 hidden units
clf = MLP(num_hid=2)
_ = clf.fit(train_data,train_labels,valid_data,valid_labels)
# validation set visualization
hid_features = clf.get_hidden(valid_data)
plot2d(hid_features,valid_labels,'valid')


# visualization for 3 hidden units
clf = MLP(num_hid=3)
_ = clf.fit(train_data,train_labels,valid_data,valid_labels)
# validation set visualization
hid_features = clf.get_hidden(valid_data)
plot3d(hid_features,valid_labels,'valid')
