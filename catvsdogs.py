import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
from PIL import Image
##############################################################################

# Load and parse Data

###############################################################################

def load_data():
	data_list = np.zeros(8, dtype = object)
	TRAIN_DIR = 'data_dogscats/train/'
	TEST_DIR = 'data_dogscats/test/'
	
	num_channels = 3
	pixel_depth = 255.0  

	TRAINING_AND_VALIDATION_SIZE_DOGS = 500 
	TRAINING_AND_VALIDATION_SIZE_CATS = 500
	TRAINING_AND_VALIDATION_SIZE_ALL  = 1000
	TRAINING_SIZE = 700  # TRAINING_SIZE + VALID_SIZE must equal TRAINING_AND_VALIDATION_SIZE_ALL
	VALID_SIZE = 300
	test_dog_size = 100
	test_cat_size = 99
	single_test_size = 1

	train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
	train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
	test_dataset = [TEST_DIR+i for i in os.listdir(TEST_DIR)]
	test_dogs =   [TEST_DIR+i for i in os.listdir(TEST_DIR) if 'dog' in i]
	test_cats =   [TEST_DIR+i for i in os.listdir(TEST_DIR) if 'cat' in i]
	train_dataset = train_dogs[:TRAINING_AND_VALIDATION_SIZE_DOGS] + train_cats[:TRAINING_AND_VALIDATION_SIZE_CATS]
	train_labels = np.array ((['dogs'] * TRAINING_AND_VALIDATION_SIZE_DOGS) + (['cats'] * TRAINING_AND_VALIDATION_SIZE_CATS))


	train_dataset = train_dataset[VALID_SIZE:VALID_SIZE+TRAINING_SIZE] 
	data_list[0] = train_dataset
	train_labels = train_labels[VALID_SIZE:VALID_SIZE+TRAINING_SIZE]
	data_list[1] = train_labels

	test_dataset = test_dogs[:test_dog_size] + test_cats[:test_cat_size]
	data_list[4] = test_dataset
	test_labels =  np.array((['dogs'] * test_dog_size) + (['cats'] * test_cat_size))
	data_list[5] = test_labels

	single_test  = test_cats[test_cat_size:]
	data_list[6]  = single_test
	single_label = np.array(['cats'])
	data_list[7] = single_label
	return data_list

def read_image(file_path):
	IMAGE_SIZE = 96
	img = Image.open(file_path) #cv2.IMREAD_GRAYSCALE
	img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
	return img

def prep_imgs(images):
    count = len(images)
    data = np.ndarray((count, 96, 96, 3), dtype=np.float32)
    pixel_depth = 255.0 

    for i, image_file in enumerate(images):
        image = read_image(image_file);
        image_data = np.array (image, dtype=np.float32);
        image_data[:,:,0] = (image_data[:,:,0].astype(float) - pixel_depth / 2) / pixel_depth
        image_data[:,:,1] = (image_data[:,:,1].astype(float) - pixel_depth / 2) / pixel_depth
        image_data[:,:,2] = (image_data[:,:,2].astype(float) - pixel_depth / 2) / pixel_depth
        
        data[i] = image_data; # image_data.T
        if i%250 == 0: print('Processed {} of {}'.format(i, count))    
    return data

def randomize(dataset, labels):
      permutation = np.random.permutation(labels.shape[0])
      shuffled_dataset = dataset[permutation,:,:,:]
      shuffled_labels = labels[permutation]
      return shuffled_dataset, shuffled_labels

def reformat(dataset, labels, image_size):
      dataset = dataset.reshape(
        (-1, image_size, image_size, 3)).astype(np.float32)
      labels = (labels=='cats').astype(np.float32); # set dogs to 0 and cats to 1
      labels = (np.arange(2) == labels[:,None]).astype(np.float32) #[0,1] = cat, [1,0] = dog
      return dataset, labels

def parse_data(data_list):
	TRAINING_SIZE = 700  
	VALID_SIZE = 300
	np.random.seed(0)
	train_normalized = prep_imgs(data_list[0])
	test_normalized = prep_imgs(data_list[4])
	single_test_normalized = prep_imgs(data_list[6])

	train_dataset, train_labels = randomize(train_normalized, data_list[1])
	test_dataset, test_labels = randomize(test_normalized, data_list[5])

	valid_dataset = train_dataset[:VALID_SIZE,:,:,:]
	valid_labels =   train_labels[:VALID_SIZE]
	train_dataset = train_dataset[VALID_SIZE:VALID_SIZE+TRAINING_SIZE,:,:,:]
	train_labels  = train_labels[VALID_SIZE:VALID_SIZE+TRAINING_SIZE]

	data_list[0], data_list[1] = reformat(train_dataset, train_labels,96 )
	data_list[2], data_list[3] = reformat(valid_dataset, valid_labels, 96)
	data_list[4], data_list[5]= reformat(test_dataset, test_labels, 96)
	data_list[6], data_list[7] = reformat(single_test_normalized, data_list[7], 96)

	return data_list

##############################################################################

# CNN functions

###############################################################################
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]
    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

def new_dropout_layer(input,
                        num_inputs, # Num. inputs from prev. layer.
                        num_outputs,
                        active_function = None):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    layer = tf.nn.dropout(layer, keep_prob)
 
    if active_function is None:
        layer =activation_function(layer, )
    return layer

def next_batch_test(data, labels, batch_size, batch_id):
	batch_data = data[batch_id:min(batch_id+ batch_size, len(data))]
	batch_labels = labels[batch_id:min(batch_id + batch_size, len(data))] 
	new_batch_id = (batch_id + batch_size) % (len(data) )

	return batch_data, batch_labels, new_batch_id

def next_batch_train(data, labels, batch_size, batch_id):
    batch_data = data[batch_id:min(batch_id+ batch_size, len(data))]
    batch_labels = labels[batch_id:min(batch_id + batch_size, len(data))] 
    new_batch_id = (batch_id + batch_size) % (len(data) - batch_size )
    
    return batch_data, batch_labels, new_batch_id

def optimize(num_iterations, data, labels, batch_size,session, x,y_true):
    # Ensure we update the global variable rather than a local copy.
    total_iterations = 0 
    # Start-time used for printing time-usage below.
    start_time = time.time()
    batch_id = 0
    for i in range(num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        batch_data, batch_labels, new_batch_id = next_batch_train(data, labels, batch_size, batch_id)


        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: batch_data,
                           y_true: batch_labels}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
        #loss_value, predicted =session.run([loss, train_prediction], feed_dict=feed_dict_train)
        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.get_accuracy(acc,batch_labels)
            print(msg.format(i + 1, acc))
        batch_id = new_batch_id

    saver.save(sess=session, save_path=save_path)


    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def calc_acc(data,labels, batch_size,session, x, y_true, batch = True):
    # Ensure we update the global variable rather than a local copy.
    # Start-time used for printing time-usage below.
    start_time = time.time()

    if not batch:
        feed_dict_train = {x: data,y_true: labels}
        
        acc =session.run(accuracy, feed_dict=feed_dict_train)
        msg = " Accuracy: {0:>6.1%}"
        print(msg.format(acc))
        
        
    else:
        
        acc_list = np.zeros(int(np.ceil(len(data)/batch_size)))
        batch_id = 0
        for i in range(int(np.ceil(len(data)/batch_size))): 
        
                # Get a batch of training examples.
                # x_batch now holds a batch of images and
                # y_true_batch are the true labels for those images.
                batch_data, batch_labels, new_batch_id = next_batch_test(data,labels, batch_size, batch_id)


                # Put the batch into a dict with the proper names
                # for placeholder variables in the TensorFlow graph.
                feed_dict_train = {x: batch_data,
                                   y_true: batch_labels}

                # Run the optimizer using this batch of training data.
                # TensorFlow assigns the variables in feed_dict_train
                # to the placeholder variables and then runs the optimizer.
                acc =session.run(accuracy, feed_dict=feed_dict_train)
                acc_list[i] = acc
                #loss_value, predicted =session.run([loss, train_prediction], feed_dict=feed_dict_train)
                # Print status every 100 iterations.
                batch_id = new_batch_id


            
                # Calculate the accuracy on the training-set.
                #acc = session.run(accuracy, feed_dict=feed_dict_train)

                # Message for printing.
                msg = "Batch number :{0:>3} , Accuracy: {1:>6.2%}"

                # Print it.
                print(msg.format(i , acc))
        total_acc = np.sum(acc_list)/len(acc_list)
        msg2 = 'Average Accuracy of {0} batchs : {1:>6.2%}'
        print(msg2.format(batch_size, total_acc))

    # calculate the whole dataset, no batch
    
        


        
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=time_dif)))





# --------- Load and Parse Data -------- #
data_list = load_data()
data_list = parse_data(data_list)

train_dataset = data_list[0]
train_labels = data_list[1]
valid_dataset= data_list[2]
valid_labels = data_list[3]
test_dataset = data_list[4]
test_labels = data_list[5]
single_test = data_list[6]
single_label = data_list[7]

# --------- CNN Model Variables -------- #
img_size = 96
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)
# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 3
# Number of classes
num_classes = 2

# Convolutional Layer 1.
filter_size1 = 10         # Convolution filters are 10 x 10 pixels.
num_filters1 = 16         # There are 16 of these filters.
# Convolutional Layer 2.
filter_size2 = 10         # Convolution filters are 10 x 10 pixels.
num_filters2 = 36         # There are 36 of these filters.
# Fully-connected layer.
fc_size = 128  

################ Build the model ################## 
###################################################

# --------- Initialize Placeholders --------- #

# shape = [input dimension: None to accept arbitrary dimension , width , height, color channels ]
# x : input images 
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

# y_true : input labels  
y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')

# if you include a dropout layer, uncomment keep_prob
# keep_prob = tf.placeholder(tf.float32)

layer_conv1, weights_conv1 = new_conv_layer(input=x,
               num_input_channels=num_channels,
               filter_size=filter_size1,
               num_filters=num_filters1,
               use_pooling=True)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
 				num_input_channels=num_filters1,
               	filter_size=filter_size2,
               	num_filters=num_filters2,
               	use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv2)

layer_fc1 = new_fc_layer(input=layer_flat,
                     num_inputs=num_features,
                     num_outputs=fc_size,
                     use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                     num_inputs=fc_size,
                     num_outputs=2,
                     use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

y_true_cls = tf.argmax(y_true, dimension=1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

saver = tf.train.Saver()

session = tf.Session()
session.run(tf.global_variables_initializer())
saver.restore(sess=session, save_path=save_path)





#optimize(150, train_dataset, train_labels,60, session, x, y_true)

calc_acc(single_test,single_label,60,session, x, y_true,False)














