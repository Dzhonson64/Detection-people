# -*- coding: utf-8 -*-
"""
## Установка TensorFlow Object Detection API
"""

!apt-get install protobuf-compiler python-pil python-lxml python-tk
!pip install Cython
!git clone https://github.com/tensorflow/models.git
!cd models/research; protoc object_detection/protos/*.proto --python_out=.
!cd models/research; export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim; python object_detection/builders/model_builder_test.py

# Commented out IPython magic to ensure Python compatibility.
import sys, os

sys.path.append('models/research')
sys.path.append('models/research/object_detection')

import numpy as np
import six.moves.urllib as urllib
import tarfile
import zipfile
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
# %matplotlib inline

"""## Загрузка предварительно обученной модели

Загружаем файл с моделью
"""

model_path = 'http://download.tensorflow.org/models/object_detection/'
model_name = 'faster_rcnn_nas_coco_2018_01_28'

os.environ['MODEL_PATH']=model_path + model_name + '.tar.gz'
os.environ['MODEL_FILE_NAME']=model_name + '.tar.gz'

!rm $MODEL_FILE_NAME
!wget $MODEL_PATH
!tar xfz $MODEL_FILE_NAME

"""Загрузка модели в память

Загрузка меток классов
"""

label_map = label_map_util.load_labelmap('models/research/object_detection/data/mscoco_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

categories

category_index

"""## Методы загрузки изображения и распознавания

Метод для загрузки изображения
"""

def load_image(image_file_name):
    image = Image.open(image_file_name)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

"""Метод для поиска объектов на одном изображении"""

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
 
      # Запуск поиска объектов
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})
 
      # Преобразование выходных данных из массивов float32 в нужный формат
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
      # print(output_dict)
      test = output_dict.copy()
      test['num_detections'] = 0
      test['detection_scores'] = ()
      test['detection_boxes'] = ()
      for i, v in enumerate(output_dict['detection_classes']):
        # print(v, i, output_dict['detection_scores'][i])
        if v == 1 and output_dict['detection_scores'][i] >= 0.6:
          test['num_detections'] += 1
          test['detection_classes'] = np.zeros((test['num_detections']), dtype='uint8')
          test['detection_scores'] = np.resize(test['detection_scores'], test['num_detections'])
          test['detection_scores'][-1] = output_dict['detection_scores'][i]
          # print(test)
          test['detection_boxes'] = np.resize(test['detection_boxes'], test['num_detections'] * 4).reshape(-1, 4)
          # print(test, output_dict['detection_boxes'].shape, test['detection_boxes'].shape)
          test['detection_boxes'][-1] = output_dict['detection_boxes'][i]
  return test

"""## Ищем пиксельные маски объектов

TensorFlow Object Detection API может находить объекты с более высокой точностью, чем прямоугольник, в границах которого находится объект. Вместо координат прямоугольника может выдаваться битовая маска. Для этого нужно использовать модели из [TensoFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), у которых в `Outputs` указано значение `Masks` (маски).
"""

# Загружаем модель с битовой маской на выходе
model_name = 'mask_rcnn_inception_v2_coco_2018_01_28'
os.environ['MODEL_PATH']=model_path + model_name + '.tar.gz'
os.environ['MODEL_FILE_NAME']=model_name + '.tar.gz'
!rm $MODEL_FILE_NAME
!wget $MODEL_PATH
!tar xfz $MODEL_FILE_NAME

# Загружаем новую модель в память
model_file_name =  model_name + '/frozen_inference_graph.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(model_file_name, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Ищем объекты и визуализируем их
def detection(image):
  output_dict = run_inference_for_single_image(image, detection_graph)
  print(output_dict['num_detections'])
  vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
  plt.figure(figsize=(12, 8))
  plt.grid(False)
  plt.imshow(image)
# for i in range(0, 1):
#   if output_dict['detection_scores'][i] > 0.3:
#     print(output_dict['detection_classes'][i],  output_dict['detection_scores'][i], category_index[output_dict['detection_classes'][i]]["name"])

! pip3 install opencv-python
! pip install opencv-contrib-python

from google.colab import drive
drive.mount('/content/drive')

import cv2
from google.colab.patches import cv2_imshow

cap = cv2.VideoCapture('drive/My Drive/Человек.mp4')
print("Did the video open? - ", cap.isOpened())
while(cap.isOpened()):
    ret, image = cap.read()
    
    detection(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2_imshow(gray)
    if cv2.waitKey(10) == 27: #Esc
        break

cap.release()
cv2.destroyAllWindows()