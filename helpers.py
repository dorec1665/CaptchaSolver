from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import numpy as np

img_folder = '/path/to/dataset'
char_to_num = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 'z':25, '*': 26}


def encode_single_sample(img_path, label, crop):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=[56,140], method='bilinear', preserve_aspect_ratio=False, antialias=False, name=None)

    if (crop == True):
        img = tf.image.crop_to_bounding_box(img, offset_height=0, offset_width=25, target_height=50, target_width=125)
        img = tf.image.resize(img, size=[56,140], method='bilinear', preserve_aspect_ratio=False, antialias=False, name=None)

    img = tf.transpose(img, perm=[1,0,2])
    label = list(map(lambda x: char_to_num[x], label))
    if len(label) == 6:
        label.append(26)
    return img.numpy(), label

def create_train_and_validation_datasets(crop=False):
    X, y = [], []
    for _, _, files in os.walk(img_folder):
        for f in files:
            label = f.split(".")[0]
            extension = f.split(".")[1]
            if extension=='png':
                img, label = encode_single_sample(img_folder+f, label, crop)
                X.append(img)
                y.append(label)
    X = np.array(X)
    y = np.array([np.array(yi) for yi in y])
    y = np.asarray(y)

    X_train, X_val, y_train, y_val = train_test_split(X.reshape(1000, 7840), y, test_size=0.01, shuffle=True, random_state=7)
    X_train, X_val = X_train.reshape(990,140,56,1), X_val.reshape(10,140,56,1)
    return X_train, X_val, y_train, y_val
