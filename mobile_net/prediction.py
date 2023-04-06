import os

import numpy as np
import tensorflow as tf
print(tf.__version__)
mobile_net_model = tf.keras.models.load_model("model_output/image_classify")


def predict_person(directory):
    count = 1
    outputs = []
    image_size = (160, 160)
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if count <= 5:
            img = tf.io.read_file(f)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, image_size)
            img.set_shape((image_size[0], image_size[1], 3))
            outputs.append(mobile_net_model.predict(np.array([img])))
        count += 1
    result = []
    for i in outputs:
        if i[0] < 0.5:
            result.append(0)
        else:
            result.append(1)

    return result


directory = '../yolo_predictions/Hayan/'
print(predict_person(directory))

directory = '../yolo_predictions/Umesh/'
print(predict_person(directory))
