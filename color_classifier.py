import numpy as np
import json
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import io
import color_config

model_file = color_config.model_file
label_file = color_config.label_file
input_layer = color_config.input_layer
output_layer = color_config.output_layer
classifier_input_size = color_config.classifier_input_size

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def load_labels(label_file):
    label = []
    with open(label_file, "r", encoding='cp1251') as ins:
        for line in ins:
            label.append(line.rstrip())

    return label

class Classifier():
    def __init__(self):
        self.graph = load_graph(model_file)
        self.labels = load_labels(label_file)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)

        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.sess.graph.finalize()  # Graph is read-only after this statement.

    def predict(self, img):
        img = img[:, :, ::-1]
        h, w = img.shape[:2]
        center_crop_size = min(w, h)
        x = int((w - center_crop_size) / 2)
        y = int((h - center_crop_size) / 2)
        img = img[y:y + center_crop_size, x:x + center_crop_size]
        img = cv2.resize(img, classifier_input_size)

        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        img /= 127.5
        img -= 1.

        results = self.sess.run(self.output_operation.outputs[0], {
            self.input_operation.outputs[0]: img
        })
        results = np.squeeze(results)

        top = 3
        top_indices = results.argsort()[-top:][::-1]
        classes = []
        for ix in top_indices:
            classes.append({"color": self.labels[ix], "prob": str(results[ix])})
        return(classes)
