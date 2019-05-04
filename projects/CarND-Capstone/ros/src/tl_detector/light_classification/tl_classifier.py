from styx_msgs.msg import TrafficLight

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import rospy
import cv2
import datetime

# from utils import label_map_util

# from utils import visualization_utils as vis_util

class TLClassifier(object):
    def __init__(self):
        sys.path.append("./../../../models/research/")
        sys.path.append("./../../../models/research/object_detection/")
        

        # from utils import label_map_util

        # from utils import visualization_utils as vis_util

        # What model to download.
        MODEL_NAME = './../../../models/research/object_detection/ssd_mobilenet_v1_coco_2017_11_17'
        MODEL_FILE = MODEL_NAME + '.tar.gz'
        # DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        MODEL_NAME = './ssd_mobilenet_v1_coco_2017_11_17'
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# /home/student/CarND-Capstone/ros/src/tl_detector/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb
        # List of the strings that is used to add correct label for each box.
        # PATH_TO_LABELS = os.path.join('/home/student/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

        NUM_CLASSES = 90

        # self.counter = 0

        # tar_file = tarfile.open(MODEL_FILE)
        # for file in tar_file.getmembers():
        #   file_name = os.path.basename(file.name)
        #   if 'frozen_inference_graph.pb' in file_name:
        #     tar_file.extract(file, os.getcwd())

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        image_np_expanded = np.expand_dims(np.zeros((10, 10, 3)), axis=0)
        # Actual detection.
        with self.detection_graph.as_default():
            self.ops = tf.get_default_graph().get_operations()
            self.all_tensor_names = {output.name for op in self.ops for output in op.outputs}
            self.tensor_dict = {}
            for key in [
                     'detection_boxes',
                      'detection_classes'
                      # , 'detection_masks'
                  ]:
                tensor_name = key + ':0'
                if tensor_name in self.all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                          tensor_name)
                # if 'detection_masks' in self.tensor_dict:
                #     # The following processing is only for single image
                #     detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
                #     detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
                #     # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                #     real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
                #     detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                #     detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                #     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                #         detection_masks, detection_boxes, image.shape[1], image.shape[2])
                #     detection_masks_reframed = tf.cast(
                #         tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                #     # Follow the convention by adding back the batch dimension
                #     self.tensor_dict['detection_masks'] = tf.expand_dims(
                #         detection_masks_reframed, 0)
            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        


        # self.run_inference_for_single_image(image_np_expanded, self.detection_graph)
          

        # label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        # category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def run_inference_for_single_image(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
              # Get handles to input and output tensors
              # ops = tf.get_default_graph().get_operations()
              # all_tensor_names = {output.name for op in ops for output in op.outputs}
              # tensor_dict = {}
              # for key in [
              #     'num_detections', 'detection_boxes', 'detection_scores',
              #     'detection_classes', 'detection_masks'
              # ]:
              #   tensor_name = key + ':0'
              #   if tensor_name in all_tensor_names:
              #     tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              #         tensor_name)
              # if 'detection_masks' in tensor_dict:
              #   # The following processing is only for single image
              #   detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
              #   detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
              #   # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
              #   real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
              #   detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
              #   detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
              #   detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              #       detection_masks, detection_boxes, image.shape[1], image.shape[2])
              #   detection_masks_reframed = tf.cast(
              #       tf.greater(detection_masks_reframed, 0.5), tf.uint8)
              #   # Follow the convention by adding back the batch dimension
              #   tensor_dict['detection_masks'] = tf.expand_dims(
              #       detection_masks_reframed, 0)
              # image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

              # Run inference
              # rospy.logwarn('--time = {0}'.format(datetime.datetime.now().time()))
              output_dict = sess.run(self.tensor_dict,
                                     feed_dict={self.image_tensor: image})
              # rospy.logwarn('--time = {0}'.format(datetime.datetime.now().time()))

              # all outputs are float32 numpy arrays, so convert types as appropriate
              # output_dict['num_detections'] = int(output_dict['num_detections'][0])
              output_dict['detection_classes'] = output_dict[
                  'detection_classes'][0].astype(np.uint8)
              output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
              # output_dict['detection_scores'] = output_dict['detection_scores'][0]
              # if 'detection_masks' in output_dict:
              #   output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        rospy.logwarn('time = {0}'.format(datetime.datetime.now().time()))
        output_dict = self.run_inference_for_single_image(image_np_expanded, self.detection_graph)
        rospy.logwarn('time = {0}'.format(datetime.datetime.now().time())) 
        lt_index = np.where(output_dict['detection_classes']==10)[0]
        # lt_index = [1]
        if len(lt_index) > 0:
            # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 1)
            # if circles is not None:
            #     # convert the (x, y) coordinates and radius of the circles to integers
            #     circles = np.round(circles[0, :]).astype("int")
             
            #     # loop over the (x, y) coordinates and radius of the circles
            #     for (x, y, r) in circles:
            #         # draw the circle in the output image, then draw a rectangle
            #         # corresponding to the center of the circle
            #         top = y - r
            #         bottom = y + r
            #         left = x - r
            #         right = x + r



            lt_index = lt_index[0]
            lt_box = output_dict['detection_boxes'][lt_index]

            im_height, im_width, _ = image.shape
            [ymin, xmin, ymax, xmax] = lt_box
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)

            # tl_box = image[int(top):int(bottom), int(left):int(right), :]

            max_all = np.max(image[int(top):int(bottom), int(left):int(right), :])
            max_red = np.max(image[int(top):int(bottom), int(left):int(right), 0])
            max_green = np.max(image[int(top):int(bottom), int(left):int(right), 1])
            max_blue = np.max(image[int(top):int(bottom), int(left):int(right), 2])

                    # max_all = np.max(image[:, :, :])
                    # max_red = np.max(image[:, :, 0])
                    # max_green = np.max(image[:, :, 1])
                    # max_blue = np.max(image[:, :, 2])
            rospy.logwarn('all={0}, red={1},green={2},blue={3}'.format(max_all, max_red, max_green, max_blue))

            if max_all == 255 and max_all == max_red:
                        if max_red == max_green:
                            # cv2.circle(image, (x, y), r, (0, 0, 0), 4)
                            # cv2.imwrite("image"+str(self.counter) + ".png", image)
                            # self.counter+=1
                            return TrafficLight.YELLOW
                        else:
                            # cv2.circle(image, (x, y), r, (0, 0, 0), 4)
                            # cv2.imwrite("image"+str(self.counter) + ".png", image)
                            # self.counter+=1
                            return TrafficLight.RED
            elif max_all == 255 and max_all == max_green:
                        # cv2.circle(image, (x, y), r, (0, 0, 0), 4)
                        # cv2.imwrite("image"+str(self.counter) + ".png", image)
                        # self.counter+=1
                        return TrafficLight.GREEN


        
        return TrafficLight.UNKNOWN
