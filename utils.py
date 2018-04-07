import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from random import shuffle
import glob
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
feature_key_path = "Data/features_060.csv"
data_folder = "Data/tfdata"
file_tail = "22"

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
    # feature_names = read_feature_names(feature_key_path)
    data_path = get_data_paths(data_folder,file_tail)

    for path in data_path:
        path_exists = tf.gfile.Exists(data_path[0])
        if not path_exists:
            print("Broken path! {}".format(path))

    #matrix feature
    # fea_name = "camera/intrinsics/matrix33"# "camera/transforms/camera_T_base/matrix44"
    # features = {fea_name: tf.FixedLenFeature([3,3], tf.float32)}
    # filename_queue = tf.train.string_input_producer(data_path,shuffle=True, num_epochs=2)
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)
    # features = tf.parse_single_example(serialized_example, features=features)
    # camera_params = features[fea_name]
    # camera_params = tf.train.shuffle_batch([camera_params], batch_size=1, capacity=12, num_threads=2, min_after_dequeue=10)
    # with tf.Session() as sess: 
    #     #initialize variables 
    #     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #     sess.run(init_op)
    #     # Create a coordinator and run all QueueRunner objects
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     ca_param =sess.run(camera_params)
    #     print(ca_param)
    #     coord.request_stop()
    #     coord.join(threads)
        # fea_name = "camera/intrinsics/matrix33"
        #     [[
        #       [ 775.71899414    0.            0.        ]
        #       [   0.          775.71899414    0.        ]
        #       [ 335.3380127   232.45100403    1.        ]
        #     ]]
    
        # fea_name = "camera/transforms/camera_T_base/matrix44"
        #     [[
        #       [-0.0210269  -0.99247903  0.120598    0.26878399]
        #       [-0.88814598 -0.0368459  -0.45808199  0.293412  ]
        #       [ 0.45908001 -0.116741   -0.88069099  0.71057898]
        #       [ 0.          0.          0.          1.        ]
        #     ]]

    #image feature
    fea_name = "grasp/0/image/encoded"
    features  = {}
    features = {fea_name: tf.FixedLenFeature([], tf.string)}
    filename_queue = tf.train.string_input_producer(data_path,shuffle=True, num_epochs=2)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=features)
    image = tf.image.decode_jpeg(features[fea_name],channels=3)#Camera RGB images are stored in JPEG format.
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = tf.reshape(image, [512,640,3])#(512, 640) random cropped to (472, 472)
    images = tf.train.shuffle_batch([image], batch_size=6, capacity=12, num_threads=2, min_after_dequeue=10)

    with tf.Session() as sess:   
        #initialize variables 
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for batch_index in range(1):
            imgs= sess.run(images)
            imgs = imgs.astype(np.uint8)
            for j in range(6):
                plt.subplot(2, 3, j + 1)
                plt.imshow(imgs[j, ...])
            plt.show()
        coord.request_stop()
        coord.join(threads)
        
if __name__ == '__main__':
    main()