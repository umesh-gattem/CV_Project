import os

import numpy as np
import tensorflow as tf
from yolo_object_detection import predict_person
print(tf.__version__)
# mobile_net_model = tf.keras.models.load_model("../mobile_net/model_output/image_classify")


def predict_persons(directory):
    count = 1
    outputs = []
    image_size = (160, 160)
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        print("hello", directory, f)
        if count <= 10 and f!= "../new_dataset/predict/.DS_Store":
            img = tf.io.read_file(f)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, image_size)
            img.set_shape((image_size[0], image_size[1], 3))
            predict_person(image_file=f, predict_actual_person=True)
            # outputs.append(mobile_net_model.predict(np.array([img])))
        count += 1
    result = []
    for i in outputs:
        result.append(np.argmax(i))
    return result


directory = '../new_dataset/predict/'
print(predict_persons(directory))

