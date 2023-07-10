import os
import numpy as np
from tensorflow import keras
import model2
import matplotlib.pyplot as plt
from helpers import create_train_and_validation_datasets

img_folder = 'path/to/dataset'

def compute_perf_metric(predictions, groundtruth):
    if predictions.shape == groundtruth.shape:
        return np.sum(predictions == groundtruth)/(predictions.shape[0]*predictions.shape[1])
    else:
        raise Exception('Error : the size of the arrays do not match. Cannot compute the performance metric')


vocabulary = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s', 't', 'u', 'v', 'w', 'x', 'y', 'z', '*'}
char_to_num = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 'z':25, '*': 26}

X_train, X_val, y_train, y_val = create_train_and_validation_datasets(crop=False)
model = model2.build_model()
exist = os.path.exists('./model_weights/checkpoint')

# Prediction with trained model
if exist:
    model.load_weights('./model_weights/model')

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )

    # prediction_model.summary()

    y_pred = prediction_model.predict(X_val)
    y_pred = keras.backend.ctc_decode(y_pred, input_length=np.ones(990)*70, greedy=True)
    y_pred = y_pred[0][0][0:10, 0:7].numpy()

    num_to_char = {'-1': 'UKN', '0': 'a', '1': 'b', '2': 'c', '3': 'd', '4': 'e', '5': 'f', '6': 'g', '7': 'h',
                   '8': 'i', '9': 'j', '10': 'k', '11': 'l', '12': 'm', '13': 'n', '14': 'o', '15': 'p', '16': 'q',
                   '17': 'r', '18': 's', '19':'t', '20':'u', '21':'v', '22':'w', '23':'x', '24':'y', '25':'z', '26': '*'}
    nrow = 1

    print("Accuracy per letter: " + str(compute_perf_metric(y_pred, y_val)))

    # Showing 10 examples of prediction
    '''imgs = []
    for i in range(0,10):
        imgs.append(X_val[i].transpose(1,0,2))
    _, axs = plt.subplots(5, 2, figsize=(20,20))
    axs = axs.flatten()

    i = 0
    for img, ax in zip(imgs, axs):
        ax.imshow(img, cmap='gray')
        ax.set_title(str(list(map(lambda x: num_to_char[str(x)], y_pred[i]))), size=40)
        i+=1

    plt.show()'''

#Training
else:
    model.summary()
    history = model.fit([X_train,y_train], epochs=200)
    model.save_weights('./model_weights/model')

