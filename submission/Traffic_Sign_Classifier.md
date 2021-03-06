
# Self-Driving Car Engineer Nanodegree

## Deep Learning




---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**



### Basic Summary of the Data Set



    Number of training examples = 39209
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43


Next, split the data train data, retaining 60% for training. Split leftout 40%
data into 50%:50%  validation and development sets. This would help reduce
overfitting to validation set. Here are the sizes of datasets after split:


    train shape (23525, 32, 32, 3)
    dev shape (7842, 32, 32, 3)
    valid shape (7842, 32, 32, 3)


Here is a visualization of frequency of examples per label in each of the
datasets: train, validation, dev, test. Visually make sure all the datasets have similar distribution of labels.


![png](output_10_1.png)



Random images visualized from training set, overlayed are the
labels:


![png](output_15_1.png)


Random images visualized from testing set, overlayed are the
labels:

![png](output_16_1.png)


----

## Step 2: Model



### Pre-process the Data Set

Images are preprocessed with histogram equalization and
min max scaling. Here are randomly chosen preprocessed images
from training set:



![png](output_22_1.png)


same preprocessing applied to test images:


![png](output_23_1.png)


Finally the images in training data are shuffled before
starting the training.

### Model Architecture


## Model Architecture

model architecture is based on LeNet-5.

**Layer 1: Convolutional.**
apply convolution with 5x5 kernel, 'Valid' padding  and stride of 2 with output shape --> (28, 28, 32)

**Activation.**
ReLU.

**Pooling.** apply max pooling with 'Valid' padding and stride of 2,
output shape --> 14x14x32.

**Layer 2: Convolutional.**   
Apply convolution with 5x5 kernel, 'Valid' padding and stride of 2 with output shape --> (10, 10, 64)

**Activation.** ReLU.

**Pooling.** Max pooling with 'Valid' padding and stride of 2,
output shape --> (5, 5, 64).

**Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
output shape --> 5*5*64 = 1600

**Layer 3: Fully Connected.**  maps 1600 inputs to 512 outputs. Dropout probability of 0.5.

**Activation.** ReLU.

**Layer 4: Fully Connected.** maps 512 inputs to 512 outputs. Dropout probability of 0.5.

**Activation.** ReLU.

**Layer 5: Fully Connected (Logits).** output logits of shape 43 corresponding to 43
traffic signs.

The code for calculating the accuracy of the model is located in the 40th cell of the Ipython notebook.


Model training is begun with following initial values of parameters:
- all the tensorflow convnet weights are initialized with truncated normal
distribution with mean of ``0.`` and stddev of ``1.``

- dropout probability of ``0.5`` is used only on fully connected layers

- EPOCHS = ``10000``

    Maximum number of epochs to train.

- BATCH_SIZE = ``256``

  maximum number of examples in training batch.

- learning rate = ``0.001``

  During training, learning rate is updated by monitoring training performance,
  using following logic (pseudocode):

      if (mean(train_accuracy_past_n_epochs) >= current_train_accuracy  
              or (best_accuracy - train_accuracy > .001)):
          learning_rate = learning_rate * .9

  This logic is meant to reduce learning rate if training accuracy
  stops improving or worsens while allowing some tolerance.


- Early stopping
  In addition, if validation accuracy exceeds target accuracy, training stops.
  Training also stops if validation accuracy worsens significantly (set to 1%)
  compared to best validation accuracy.






After 62 epochs, training accuracy of 0.999, validation accuracy of 0.991, with AvgEpochTime 21.95 s, and  TotalTime 22.68 min, is achieved. The model is trained with NVIDIA 740M gpu.


On additional held out data, previously referred to as development set, an accuracy of 0.989 is achieved.
At this point, with enough validation confidence, evaluate model on test images.

On the test set, an accuracy of  0.948 is achieved.


here are some visualizations of activations for convolutional layers:

---

convolution layer 1 visualization from test data:



![png](output_35_1.png)

---

convolution layer 2 visualization from test data



![png](output_36_1.png)


---

## Step 3: Test a Model on New Images

### Load and Output the Images

5 new German traffic signs images downloaded from the internet:

![png](output_41_0.png)

After applying same preprocessing to the new images:


![png](output_44_0.png)


Top 5 predictions on new images:

![png](output_46_0.png)



![png](output_46_1.png)



![png](output_46_2.png)



![png](output_46_3.png)



![png](output_46_4.png)


### Analyze Performance

The model failed to guess all 5 new traffic signs. In case of
one traffic sign - "speed-limit-30" - the prediction corresponding to second highest probability is correct. This performance does not compare well with the test set performance of 94%.

A few characteristics that might be affecting new image prediction performance are:

- It is possible that resizing images from the Internet
skews the traffic signs in ways that are not seen by the
model.

- drastically different image background contrast.

- angle of traffic signs

- these images also look way more jittery than the training
images.

---

## Step 4: Visualize the Neural Network's State with Test Images



### Visualize the activations of convnet layers on new images


    stop.jpeg



![png](output_53_1.png)



![png](output_53_2.png)




    speed-limit-60.jpeg



![png](output_54_1.png)



![png](output_54_2.png)


Above visualizations show that the activations have learnt relevant features from images.
