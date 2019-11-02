import os
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def load_data(dir_name):
    
    images = []
    labels = []
    
    dirs = [ d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d)) ]

    for d in dirs:
        
        label_dir = os.path.join(dir_name, d)
        files = [ os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.ppm') ]

        for f in files:
            images.append(cv2.imread(f))
            labels.append(int(d))

    return np.array(images), np.array(labels)


def plot_data(num, images):

    ind = [ random.randint(0, len(images)) for i in range(num) ]
    print(ind)
    for i, ent in enumerate(ind):
        plt.subplot(1, 4, i+1)
        plt.axis('off')
        plt.imshow(images[ent])
        plt.subplots_adjust(wspace = 0.5)
    plt.show()

def plot_all_labels(labels, images):
    
    unique_labels = set(labels)
    for i, label in enumerate(unique_labels):
        
        plt.subplot(8, 8, i+1)
        plt.axis('off')
        plt.imshow(images[ labels.tolist().index(label) ])
        plt.title("{0} -> {1}".format(label, labels.tolist().count(label)))
        plt.subplots_adjust(wspace = 0.5)
        plt.subplots_adjust(hspace = 1.0)
    plt.show()

def resize_images(images, size):

    resized = [ cv2.resize(image, size) for image in images]
    return resized

def rgb2gray(images):

    gray = [ cv2.cvtColor(image, cv2.COLOR_RGB2GRAY ) for image in images]
    return gray

def create_model(x_train, y_train, x_test, y_test):

    x = tf.placeholder( dtype = tf.float32, shape = [None, 28, 28])
    y = tf.placeholder( dtype = tf.int32, shape = [None])

    images_flat = tf.contrib.layers.flatten(x)
    logits2 = tf.contrib.layers.fully_connected(images_flat, 256, tf.nn.relu)
    logits1 = tf.contrib.layers.fully_connected(logits2, 128, tf.nn.relu)
    logits = tf.contrib.layers.fully_connected(logits1, 62, tf.nn.relu)
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
    optimizer = tf.train.AdamOptimizer( learning_rate = 0.001).minimize(loss)
    
    correct_pred = tf.argmax(logits, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())
    writer.flush()

    tf.set_random_seed(1234)

    with tf.Session() as sess:
        
        p = np.random.permutation(len(x_train))
        x_t = x_train.copy()
        y_t = y_train.copy()

        x_train = [x_t[i] for i in p]
        y_train = [y_t[i] for i in p]
        
        p = np.random.permutation(len(x_test))
        x_t = x_test.copy()
        y_t = y_test.copy()

        x_test = [x_t[i] for i in p]
        y_test = [y_t[i] for i in p]

        sess.run(tf.global_variables_initializer())
        for i in range(401):
            
            print("Epoch number", i)
            _, loss_value = sess.run( [optimizer, loss], feed_dict = {x : x_train, y : y_train })
            if i % 10 == 0:
                print("Loss for epoch number {0} is {1}".format(i, loss_value))

        pred = sess.run( [ correct_pred ], feed_dict = {x : x_test } ) [0]
       
        accuracy = sum( [ int(y_ == y) for y_, y in zip(pred, y_test) ])
        print("The accuracy of the classifier is {0}".format(accuracy/len(y_test)))

        ra = [ np.random.randint(0, len(x_test)) for i in range(0, 16)]
        for i, im in enumerate(ra):

            plt.subplot(4, 4, i+1)
            plt.imshow(x_test[im])
            plt.title("T {0}, P {1}".format( y_test[im], pred[im]))
            plt.subplots_adjust(wspace = 0.5)
            plt.subplots_adjust(hspace = 1.0)
        plt.show()

        

ROOT_PATH = "/Users/chanddra/Columbia/TF/data"
TRAIN_PATH = os.path.join(ROOT_PATH, "Training")
TEST_PATH = os.path.join(ROOT_PATH, "Testing")

if __name__ == "__main__":
    x_train, y_train = load_data(TRAIN_PATH)
    x_test, y_test = load_data(TEST_PATH)

    x_train = resize_images(x_train, (28, 28))
    x_test = resize_images(x_test, (28, 28))
    
    x_train = rgb2gray(x_train)
    x_test = rgb2gray(x_test)
    
    print("Creating the model")
    create_model(x_train, y_train, x_test, y_test)

