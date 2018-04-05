import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from random import shuffle
import glob
import os
import csv
from pprint import pprint
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
feature_key_path = "Data/features_060.csv"
#data_folder = "Data/tfdata"
#file_tail = "22"
data_folder = "Data/tfdata1"
file_tail = "34"

def read_feature_names(csv_path):
    """
    Input:
        csv_path: a string which is the relative path to descriptive csv file
    Output:
        feature_names: a list of feature names 
    """
    # read in feature names
    feature_names = []
    with open(csv_path, newline='') as csvfile:
        names_reader = csv.reader(csvfile)
        row = -1
        for name in names_reader:
            row +=1
            if row == 0 or row ==1:
                continue
            feature_names.append(*name)
    return feature_names

def get_data_paths(data_folder, file_tail):
    """
    Inputs:
        data_folder: a string, relative path to data folder
        file_tail: a string, common tail string to data files
    Output:
        data_path: a list of sorted path strings to data files
    """
    # get data file names
    #could use glob function for simplicity
    file_names = os.listdir(data_folder)
    filtered_filenames = []
    for file_name in file_names:
        if(file_name.endswith(file_tail)):
            filtered_filenames.append(file_name)
    file_names = sorted(filtered_filenames, key=lambda name: int(name[-11:-9]))

    # get relative data path based on file names
    data_path = []
    path_prefix = data_folder + "/"
    for file_name in file_names:
        data_path.append(path_prefix + file_name)
        #print(path_prefix + file_name)
    return data_path

def main():    
    feature_names = read_feature_names(feature_key_path)
    #data_path = get_data_paths(data_folder,file_tail)
    pprint(feature_names)
    return

    for path in data_path:
        path_exists = tf.gfile.Exists(data_path[0])
        if not path_exists:
            print("Broken path! {}".format(path))
    print(data_path)
    # data_path = ['Data/tfdata/grasping_dataset_060.tfrecord-00000-of-00022']
    fea_name = "present/image/encoded"#"post_drop/image/encoded"#"grasp/1/image/encoded"#"grasp/image/encoded"#"grasp/0/image/encoded"
    features = {fea_name: tf.FixedLenFeature([], tf.string)}
    
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(data_path,shuffle=True, num_epochs=2)
    #print(type(filename_queue))
    #print(filename_queue.shapes)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    #Approach 1 :decode from parsed features
    features = tf.parse_single_example(serialized_example, features=features)
    image = tf.image.decode_jpeg(features[fea_name],channels=3)#Camera RGB images are stored in JPEG format.
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    
    #Approach 2 : decode from serialized_example directly. Any difference in this case?
    #image = tf.image.decode_jpeg(serialized_example, channels=3)

    image = tf.reshape(image, [512,640,3])#(512, 640) random cropped to (472, 472)
    # image = tf.transpose(image, [1,2,0])

    #batch method 1: shuffle batch
    images = tf.train.shuffle_batch([image], batch_size=6, capacity=12, num_threads=2, min_after_dequeue=10)
    #batch method 2: batch
    # images = tf.train.batch([image], batch_size=6, capacity=12, num_threads=2)

    with tf.Session() as sess:   
        #initialize variables 
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for batch_index in range(30):
            img= sess.run(images)
            img = img.astype(np.uint8)
            # plt.imshow(img)
            for j in range(6):
                plt.subplot(2, 3, j + 1)
                plt.imshow(img[j, ...])
            plt.show()
        # Stop the threads
        coord.request_stop()
        # # Wait for threads to stop
        coord.join(threads)
        
if __name__ == '__main__':
    main()