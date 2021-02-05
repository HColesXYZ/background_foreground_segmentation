import sys
sys.path.append("..")
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from bfseg.utils.datasets import load_data
import datetime
import tensorflow as tf
from tensorflow import keras
from bfseg.cl_models import BaseCLModel


class EWC(BaseCLModel):
  r"""EWC model.

  Args:
    run (sacred.run.Run): Object identifying the current sacred run.
    root_output_dir (str): Path to the folder that will contain the experiment
      logs and the saved models.
    fisher_params_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset):
      Dataset from which the Fisher parameters should be computed. Assuming that
      the current model is being used on the second of two tasks, this dataset
      should be the one used to train the model on the first task.
    lambda_ewc (float): Regularization hyperparameter used to weight
      the loss. Valid values are between 0 and 1. In particular, the loss is
      computed as:
        (1 - lambda_ewc) * loss_ce + lambda_ewc * consolidation_loss, where
      - loss_ce is the cross-entropy loss computed on the current task;
      - consolidation_loss is the regularization loss on the parameters from the
        previous task.    
  
  References:
    - "Overcoming catastrophic forgetting in neural networks"
       (https://arxiv.org/abs/1612.00796).
  """

  def __init__(self, run, root_output_dir, fisher_params_ds, lambda_ewc):
    super(EWC, self).__init__(run=run, root_output_dir=root_output_dir)

    self._lambda_ewc = lambda_ewc
    self.model_save_dir = os.path.join('../experiments', self.config.exp_name,
                                       'lambda' + str(self._lambda_ewc),
                                       'saved_model')

    print(
        "NOTE: It is assumed that the test dataset coincides with the previous "
        "task!")
    self.create_old_params()
    self.create_fisher_params(fisher_params_ds)

  def create_old_params(self):
    """ Keep old weights of the model"""
    self.old_params = []
    for param in self.new_model.trainable_weights:
      old_param_name = param.name.replace(':0', '_old')
      self.old_params.append(
          tf.Variable(param, trainable=False, name=old_param_name))

  def create_fisher_params(self, dataset):
    """ Compute sqaured fisher information, representing relative importance"""
    self.fisher_params = []
    grads_list = [
    ]  # list of list of gradients, outer: for different batches, inner: for different network parameters
    for step, (x, y, m) in enumerate(dataset):
      if step > 40:
        break
      with tf.GradientTape() as tape:
        [_, pred_y] = self.new_model(x, training=True)
        log_y = tf.math.log(pred_y)
        y = tf.cast(y, log_y.dtype)
        log_likelihood = tf.reduce_sum(y * log_y[:, :, :, 1:2] +
                                       (1 - y) * log_y[:, :, :, 0:1],
                                       axis=[1, 2, 3])
      grads = tape.gradient(log_likelihood, self.new_model.trainable_weights)
      grads_list.append(grads)
    fisher_params = []
    fisher_param_names = [
        param.name.replace(':0', '_fisher')
        for param in self.new_model.trainable_weights
    ]
    ## compute expectation
    for i in range(len(fisher_param_names)):
      single_fisher_param_list = [tf.square(param[i]) for param in grads_list]
      fisher_params.append(
          tf.reduce_mean(tf.stack(single_fisher_param_list, 0), 0))
    for param_name, param in zip(fisher_param_names, fisher_params):
      self.fisher_params.append(
          tf.Variable(param, trainable=False, name=param_name))

  def compute_consolidation_loss(self):
    """ Compute weight regularization term """
    losses = []
    for i, param in enumerate(self.new_model.trainable_weights):
      losses.append(
          tf.reduce_sum(self.fisher_params[i] *
                        (param - self.old_params[i])**2))
    return tf.reduce_sum(losses)

  def build_loss_and_metric(self):
    """ Add loss criteria and metrics"""
    self.loss_ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.loss_mse = keras.losses.MeanSquaredError()
    self.loss_tracker = keras.metrics.Mean('loss', dtype=tf.float32)
    self.loss_ce_tracker = keras.metrics.Mean('loss_ce', dtype=tf.float32)
    self.loss_mse_tracker = keras.metrics.Mean('loss_weights', dtype=tf.float32)
    self.acc_metric = keras.metrics.Accuracy('accuracy')

  def train_step(self, train_x, train_y, train_m, step):
    """ Training on one batch:
            Compute masked cross entropy loss(true label, predicted label)
            + Weight consolidation loss(new weights, old weights)
            update losses & metrics
        """
    with tf.GradientTape() as tape:
      [_, pred_y] = self.new_model(train_x, training=True)
      pred_y_masked = tf.boolean_mask(pred_y, train_m)
      train_y_masked = tf.boolean_mask(train_y, train_m)
      output_loss = self.loss_ce(train_y_masked, pred_y_masked)
      consolidation_loss = self.compute_consolidation_loss()
      loss = (1 - self._lambda_ewc
             ) * output_loss + self._lambda_ewc * consolidation_loss
    grads = tape.gradient(loss, self.new_model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.new_model.trainable_weights))
    pred_y = tf.math.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, train_m)
    self.acc_metric.update_state(train_y_masked, pred_y_masked)

    if self.config.tensorboard_write_freq == "batch":
      with self.train_summary_writer.as_default():
        tf.summary.scalar('loss,lambda=' + str(self._lambda_ewc),
                          loss,
                          step=step)
        tf.summary.scalar('loss_ce,lambda=' + str(self._lambda_ewc),
                          output_loss,
                          step=step)
        tf.summary.scalar('loss_weights,lambda=' + str(self._lambda_ewc),
                          consolidation_loss,
                          step=step)
        tf.summary.scalar('accuracy,lambda=' + str(self._lambda_ewc),
                          self.acc_metric.result(),
                          step=step)
      self.acc_metric.reset_states()
    elif self.config.tensorboard_write_freq == "epoch":
      self.loss_tracker.update_state(loss)
      self.loss_ce_tracker.update_state(output_loss)
      self.loss_mse_tracker.update_state(consolidation_loss)
    else:
      raise Exception("Invalid tensorboard_write_freq: %s!" %
                      self.config.tensorboard_write_freq)

  def test_step(self, test_x, test_y, test_m):
    """ Validating/Testing on one batch
            update losses & metrics
        """
    [_, pred_y] = self.new_model(test_x, training=True)
    pred_y_masked = tf.boolean_mask(pred_y, test_m)
    test_y_masked = tf.boolean_mask(test_y, test_m)
    output_loss = self.loss_ce(test_y_masked, pred_y_masked)
    consolidation_loss = self.compute_consolidation_loss()
    loss = (1 - self._lambda_ewc
           ) * output_loss + self._lambda_ewc * consolidation_loss
    pred_y = tf.math.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, test_m)
    # Update val/test metrics
    self.loss_tracker.update_state(loss)
    self.loss_ce_tracker.update_state(output_loss)
    self.loss_mse_tracker.update_state(consolidation_loss)
    self.acc_metric.update_state(test_y_masked, pred_y_masked)

  def on_epoch_end(self, epoch, val_ds):
    """save models after every several epochs
        """
    if (epoch + 1) % self.config.model_save_freq == 0:
      # compute validation accuracy as part of the model name
      for val_x, val_y, val_m in val_ds:
        self.test_step(val_x, val_y, val_m)
      self.new_model.save(
          os.path.join(
              self.model_save_dir, 'model.' + str(epoch) + '-' +
              str(self.acc_metric.result().numpy())[:5] + '.h5'))
      self.loss_tracker.reset_states()
      self.loss_ce_tracker.reset_states()
      self.loss_mse_tracker.reset_states()
      self.acc_metric.reset_states()

  def write_to_tensorboard(self, summary_writer, step):
    """
            write losses and metrics to tensorboard
        """
    with summary_writer.as_default():
      tf.summary.scalar('loss,lambda=' + str(self._lambda_ewc),
                        self.loss_tracker.result(),
                        step=step)
      tf.summary.scalar('loss_ce,lambda=' + str(self._lambda_ewc),
                        self.loss_ce_tracker.result(),
                        step=step)
      tf.summary.scalar('loss_weights,lambda=' + str(self._lambda_ewc),
                        self.loss_mse_tracker.result(),
                        step=step)
      tf.summary.scalar('accuracy,lambda=' + str(self._lambda_ewc),
                        self.acc_metric.result(),
                        step=step)
    self.loss_tracker.reset_states()
    self.loss_ce_tracker.reset_states()
    self.loss_mse_tracker.reset_states()
    self.acc_metric.reset_states()
