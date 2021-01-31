import sys
sys.path.append("..")
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from bfseg.sacred_utils import get_observer
from bfseg.settings import TMPDIR
from sacred import Experiment
import segmentation_models as sm
from shutil import make_archive

ex = Experiment()
ex.observers.append(get_observer())


@ex.config
def seg_experiment_default_config():
  r"""Default configuration for base segmentation experiments.
  - batch_size (int): Batch size.
  - num_training_epochs (int): Number of training epochs
  - image_w (int): Image width.
  - image_f (int): Image height.
  - exp_name (str): Name of the current experiment.
  - backbone (str): Name of the backbone of the U-Net architecture.
  - learning_rate (float): Learning rate.
  - train_dataset (str): Name of the training dataset.
  - test_dataset (str): Name of the test dataset.
  - train_scene (str): Scene type of the training dataset. Valid values are:
      None, "kitchen", "bedroom".
  - test_scene (str): Scene type of the test dataset. Valid values are: None,
      "kitchen", "bedroom".
  - pretrained_dir (str): Directory containing the pretrained model weights.
  - tensorboard_write_freq (str): Tensorboard writing frequency. Valid values
      are "epoch", "batch".
  - model_save_freq (int): Frequency (in epochs) for saving models.
  """
  batch_size = 8
  num_training_epochs = 3
  #TODO (fmilano): Retrieve from first training sample.
  image_w = 640
  image_h = 480

  exp_name = "exp_stage1"
  backbone = "vgg16"
  learning_rate = 1e-5
  train_dataset = "BfsegCLAMeshdistLabels"
  test_dataset = "NyuDepthV2Labeled"
  train_scene = None
  test_scene = None
  pretrained_dir = None
  tensorboard_write_freq = "batch"
  model_save_freq = 1


class BaseSegExperiment():
  """
    Base class to specify an experiment.
    An experiment is a standalone class that supports:
    - Loading training data
    - Creating Models that can be trained
    - Creating experiment specific loss functions
    - Creating tensorboard writer for visualization
    - Training the model
    """

  def __init__(self):
    self.log_dir = os.path.join(TMPDIR, ex.current_run.config['exp_name'],
                                'logs')
    self.model_save_dir = os.path.join(TMPDIR,
                                       ex.current_run.config['exp_name'],
                                       'models')
    self.optimizer = keras.optimizers.Adam(
        ex.current_run.config['learning_rate'])

  def make_dirs(self):
    try:
      os.makedirs(os.path.join(self.log_dir, 'train'))
      os.makedirs(os.path.join(self.log_dir, 'val'))
      os.makedirs(os.path.join(self.log_dir, 'test'))
      os.makedirs(self.model_save_dir)
    except os.error:
      pass

  @tf.function
  def preprocess_nyu(self, image, label):
    """ Preprocess NYU dataset:
            Normalize images: `uint8` -> `float32`.
            label: 1 if belong to background, 0 if foreground
            create all-True mask since nyu labels are all known
        """
    mask = tf.not_equal(label, -1)  #all true
    label = tf.expand_dims(label, axis=2)
    image = tf.cast(image, tf.float32) / 255.
    return image, label, mask

  @tf.function
  def preprocess_cla(self, image, label):
    """ Preprocess our auto-labeled CLA dataset:
            It consists of three labels (0,1,2) where all classes that belong to the background 
            (e.g. floor, wall, roof) are assigned the '2' label. Foreground has assigned the 
            '0' label and unknown the '1' label,
            Let CLA label format to be consistent with NYU
            label: 1 if belong to background, 0 if foreground / unknown(does not matter since we are using masked loss)
            Mask element is True if it's known (label '0' or '2'), mask is used to compute masked loss
        """
    mask = tf.squeeze(tf.not_equal(label, 1))
    label = tf.cast(label == 2, tf.uint8)
    image = tf.cast(image, tf.float32)
    return image, label, mask

  def load_data(self, dataset_name, mode, batch_size, scene_type):
    """ Create a dataloader given:
            name of dataset: NyuDepthV2Labeled/ BfsegCLAMeshdistLabels,
            mode: train/ val/ test
            type of scene: None/ kitchen/ bedroom
        """
    if (dataset_name == 'NyuDepthV2Labeled'):
      if (scene_type == None):
        name = 'full'
      elif (scene_type == "kitchen"):
        name = 'train'
      elif (scene_type == "bedroom"):
        name = 'test'
      else:
        raise Exception("Invalid scene type: %s!" % scene_type)
    elif (dataset_name == 'BfsegCLAMeshdistLabels'):
      name = 'fused'
    else:
      raise Exception("Dataset %s not found!" % dataset_name)
    if (mode == 'train'):
      split = name + '[:80%]'
      shuffle = True
    else:
      split = name + '[80%:]'
      shuffle = False
    ds, info = tfds.load(
        dataset_name,
        split=split,
        shuffle_files=shuffle,
        as_supervised=True,
        with_info=True,
    )
    if (dataset_name == 'NyuDepthV2Labeled'):
      ds = ds.map(self.preprocess_nyu,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif (dataset_name == 'BfsegCLAMeshdistLabels'):
      ds = ds.map(self.preprocess_cla,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    if (mode == 'train'):
      ds = ds.shuffle(int(info.splits[name].num_examples * 0.8))
    ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def load_datasets(self):
    """create 3 dataloaders for training, validation and testing """
    train_dataset = ex.current_run.config['train_dataset']
    train_scene = ex.current_run.config['train_scene']
    test_dataset = ex.current_run.config['test_dataset']
    test_scene = ex.current_run.config['test_scene']
    batch_size = ex.current_run.config['batch_size']
    train_ds = self.load_data(dataset_name=train_dataset,
                              mode='train',
                              batch_size=batch_size,
                              scene_type=train_scene)
    val_ds = self.load_data(dataset_name=train_dataset,
                            mode='val',
                            batch_size=batch_size,
                            scene_type=train_scene)
    test_ds = self.load_data(dataset_name=test_dataset,
                             mode='test',
                             batch_size=batch_size,
                             scene_type=test_scene)
    return train_ds, val_ds, test_ds

  def create_old_params(self):
    """ Keep old weights of the model"""
    pass

  def create_fisher_params(self, dataset):
    """ Compute sqaured fisher information, representing relative importance"""
    pass

  def compute_consolidation_loss(self):
    """ Compute weight regularization term """
    pass

  def build_model(self):
    """ Build models"""
    self.encoder, self.model = sm.Unet(
        backbone_name=ex.current_run.config['backbone'],
        input_shape=(ex.current_run.config['image_h'],
                     ex.current_run.config['image_w'], 3),
        classes=2,
        activation='sigmoid',
        weights=ex.current_run.config['pretrained_dir'],
        encoder_freeze=False)
    self.new_model = keras.Model(
        inputs=self.model.input,
        outputs=[self.encoder.output, self.model.output])

  def build_tensorboard_writer(self):
    """ Create tensorboard writers"""
    self.train_summary_writer = tf.summary.create_file_writer(self.log_dir +
                                                              "/train")
    self.val_summary_writer = tf.summary.create_file_writer(self.log_dir +
                                                            "/val")
    self.test_summary_writer = tf.summary.create_file_writer(self.log_dir +
                                                             "/test")

  def build_loss_and_metric(self):
    """ Add loss criteria and metrics"""
    self.loss_ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.loss_tracker = keras.metrics.Mean('loss', dtype=tf.float32)
    self.acc_metric = keras.metrics.Accuracy('accuracy')

  def train_step(self, train_x, train_y, train_m, step):
    """ Training on one batch:
            Compute masked cross entropy loss(true label, predicted label),
            update losses & metrics
        """
    with tf.GradientTape() as tape:
      [_, pred_y] = self.new_model(train_x, training=True)
      pred_y_masked = tf.boolean_mask(pred_y, train_m)
      train_y_masked = tf.boolean_mask(train_y, train_m)
      loss = self.loss_ce(train_y_masked, pred_y_masked)
    grads = tape.gradient(loss, self.new_model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.new_model.trainable_weights))
    pred_y = tf.math.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, train_m)
    self.acc_metric.update_state(train_y_masked, pred_y_masked)

    tensorboard_write_freq = ex.current_run.config['tensorboard_write_freq']
    if (tensorboard_write_freq == "batch"):
      with self.train_summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('accuracy', self.acc_metric.result(), step=step)
      self.acc_metric.reset_states()
    elif (tensorboard_write_freq == "epoch"):
      self.loss_tracker.update_state(loss)
    else:
      raise Exception("Invalid tensorboard_write_freq: %s!" %
                      tensorboard_write_freq)

  def test_step(self, test_x, test_y, test_m):
    """ Validating/Testing on one batch
            update losses & metrics
        """
    [_, pred_y] = self.new_model(test_x, training=False)
    pred_y_masked = tf.boolean_mask(pred_y, test_m)
    test_y_masked = tf.boolean_mask(test_y, test_m)
    loss = self.loss_ce(test_y_masked, pred_y_masked)
    pred_y = keras.backend.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, test_m)
    # Update val/test metrics
    self.loss_tracker.update_state(loss)
    self.acc_metric.update_state(test_y_masked, pred_y_masked)

  def on_epoch_end(self, epoch, val_ds):
    """ save models after every several epochs
        """
    if ((epoch + 1) % ex.current_run.config['model_save_freq'] == 0):
      # compute validation accuracy as part of the model name
      for val_x, val_y, val_m in val_ds:
        self.test_step(val_x, val_y, val_m)
      self.new_model.save(
          os.path.join(
              self.model_save_dir, 'model.' + str(epoch) + '-' +
              str(self.acc_metric.result().numpy())[:5] + '.h5'))
      self.loss_tracker.reset_states()
      self.acc_metric.reset_states()

  def write_to_tensorboard(self, summary_writer, step):
    """
            write losses and metrics to tensorboard
        """
    with summary_writer.as_default():
      tf.summary.scalar('loss', self.loss_tracker.result(), step=step)
      tf.summary.scalar('accuracy', self.acc_metric.result(), step=step)
    self.loss_tracker.reset_states()
    self.acc_metric.reset_states()

  def training(self, train_ds, val_ds, test_ds):
    """
        train for assigned epochs, and validate & test after each epoch/batch,
        save models after every several epochs
        """
    step = 0
    tensorboard_write_freq = ex.current_run.config['tensorboard_write_freq']
    for epoch in range(ex.current_run.config['num_training_epochs']):
      print("\nStart of epoch %d" % (epoch,))
      if (tensorboard_write_freq == "batch"):
        for train_x, train_y, train_m in train_ds:
          self.train_step(train_x, train_y, train_m, step)
          for val_x, val_y, val_m in val_ds:
            self.test_step(val_x, val_y, val_m)
          self.write_to_tensorboard(self.val_summary_writer, step)
          for test_x, test_y, test_m in test_ds:
            self.test_step(test_x, test_y, test_m)
          self.write_to_tensorboard(self.test_summary_writer, step)
          step += 1
        if epoch == 0:
          print("There are %d batches in the training dataset" % step)
      elif (tensorboard_write_freq == "epoch"):
        for train_x, train_y, train_m in train_ds:
          self.train_step(train_x, train_y, train_m, step)
        self.write_to_tensorboard(self.train_summary_writer, epoch)
        for val_x, val_y, val_m in val_ds:
          self.test_step(val_x, val_y, val_m)
        self.write_to_tensorboard(self.val_summary_writer, epoch)
        for test_x, test_y, test_m in test_ds:
          self.test_step(test_x, test_y, test_m)
        self.write_to_tensorboard(self.test_summary_writer, epoch)
      self.on_epoch_end(epoch, val_ds)


@ex.main
def run(_run, batch_size, num_training_epochs, image_w, image_h, exp_name,
        backbone, learning_rate, train_dataset, test_dataset, train_scene,
        test_scene, pretrained_dir, tensorboard_write_freq, model_save_freq):
  """ Whole Training pipeline"""
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  print("Current time is " + current_time)
  seg_experiment = BaseSegExperiment()
  # Set up the experiment.
  seg_experiment.make_dirs()
  seg_experiment.build_model()
  seg_experiment.build_tensorboard_writer()
  seg_experiment.build_loss_and_metric()
  train_ds, val_ds, test_ds = seg_experiment.load_datasets()
  # Run the training.
  seg_experiment.training(train_ds, val_ds, test_ds)
  # Save the data to sacred.
  path_to_archive_model = make_archive(seg_experiment.model_save_dir, 'zip',
                                       seg_experiment.model_save_dir)
  _run.add_artifact(path_to_archive_model)


if __name__ == "__main__":
  ex.run_commandline()
