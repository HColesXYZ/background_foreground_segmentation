#!/usr/bin/env python
"""
  ROS node that segments the camera image using saved tensorflow models
"""
import sys
import rospy
import message_filters
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from zipfile import ZipFile
import os
import time
# Load model to predict foreground/background
import numpy as np
import tensorflow as tf
import gdown

from bfseg.settings import TMPDIR

tf.executing_eagerly()

def load_gdrive_file(file_id,
                     ending='',
                     output_folder=os.path.expanduser('~/.keras/datasets')):
  """Downloads files from google drive, caches files that are already downloaded."""
  filename = '{}.{}'.format(file_id, ending) if ending else file_id
  filename = os.path.join(output_folder, filename)
  if not os.path.exists(filename):
    gdown.download('https://drive.google.com/uc?id={}'.format(file_id),
                   filename,
                   quiet=False)
  return filename

def callback(pred_func, img_pubs, *image_msgs):
  """ Gets executed every time we get an image from the camera"""
  # Set headers for camera info
  startTime = time.time()

  for msg, pub in zip(image_msgs, img_pubs):
      img = np.frombuffer(msg.data, dtype=np.uint8)
      #img = np.rot90(img, k=1)
      if rospy.get_param('~pseudo_rgb'):
        img = img.reshape(msg.height, msg.width)
        img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)
      else:
        img = img.reshape(msg.height, msg.width, 3)

      # Convert BGR to RGB
      if 'bgr' in msg.encoding.lower():
          img = img[:, :, [2, 1, 0]]

      # resize to common input format
      img = tf.image.convert_image_dtype(tf.convert_to_tensor(img), tf.float32)
      img = tf.image.resize(img, (rospy.get_param('~input_height'), rospy.get_param('~input_width')))

      # Predict
      pred = pred_func(tf.expand_dims(img, 0))

      # Resize prediction
      prediction = tf.image.resize(pred[..., tf.newaxis], (msg.height, msg.width), tf.image.ResizeMethod.BILINEAR)

      # Convert to numpy
      prediction = prediction.numpy().astype('uint8')

      # Create and publish image message
      img_msg = Image()
      img_msg.header = msg.header
      img_msg.height = msg.height
      img_msg.width = msg.width
      img_msg.step = msg.width
      img_msg.data = prediction.flatten().tolist()
      img_msg.encoding = "mono8"
      pub.publish(img_msg)

  timeDiff = time.time() - startTime
  print("published segmented images in {:.4f}s, {:.4f} FPs".format(
      timeDiff, 1 / timeDiff))


def main_loop():
  rospy.init_node('inference_node')
  image_subscribers = [
      message_filters.Subscriber(topic, Image)
      for topic in rospy.get_param('~image_topics')
  ]
  img_pubs = [
      rospy.Publisher(topic, Image, queue_size=10)
      for topic in rospy.get_param('~segmentation_output_topics')
  ]
  # load the  model
  ZipFile(load_gdrive_file(rospy.get_param('~model_gdrive_id'),
                           ending='zip')).extractall(
                               os.path.join(TMPDIR, 'segmentation_model'))
  model = tf.saved_model.load(os.path.join(TMPDIR, 'segmentation_model'))

  @tf.function
  def pred_func(batch):
    # predict batch of images
    return tf.squeeze(tf.nn.softmax(model(batch), axis=-1)[..., 1] * 255)

  synchronizer = message_filters.ApproximateTimeSynchronizer(image_subscribers, 10, 0.1)
  synchronizer.registerCallback(lambda *x: callback(pred_func, img_pubs, *x))
  rospy.spin()

if __name__ == '__main__':
  try:
    main_loop()
  except KeyboardInterrupt as e:
    pass
  print("exiting")
