# coding=utf-8
"""
Created on Thu Jul  9 19:25:30 2020

@author: cm
"""


import os
import csv
import random
import collections
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import metrics as contrib_metrics
from classifier_multi_label_textcnn import modeling
from classifier_multi_label_textcnn import optimization
from classifier_multi_label_textcnn import tokenization
from classifier_multi_label_textcnn.hyperparameters import Hyperparamters as hp
from classifier_multi_label_textcnn.utils import load_csv


def label2id(label):
    return hp.dict_label2id[str(label)]


def id2label(index):
    return hp.dict_id2label[str(index)]


def read_csv(input_file):
    """Reads a tab separated value file."""
    df = load_csv(input_file,header=0).fillna('|')
    jobcontent = df['content'].tolist()
    jlabel = df.loc[:,hp.label_vocabulary].values
    lines = [[jlabel[i],jobcontent[i]] for i in range(len(jlabel)) if type(jobcontent[i])==str]
    return lines


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               guid=None,
               example_id=None,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.example_id = example_id
    self.guid = guid
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def __init__(self, use_spm, do_lower_case):
    super(DataProcessor, self).__init__()
    self.use_spm = use_spm
    self.do_lower_case = do_lower_case

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_csv(cls,input_file):
    """Reads a tab separated value file."""
    df = load_csv(input_file,header=0).fillna('|')
    jobcontent = df['content'].tolist()
    jlabel = df.loc[:,hp.label_vocabulary].values
    lines = [[jlabel[i],jobcontent[i]] for i in range(len(jlabel)) if type(jobcontent[i])==str]
    print('Length of data:',len(lines))
    return lines


class ClassifyProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(            
            self._read_csv(os.path.join(data_dir, hp.train_data)), "train")
        
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, hp.test_data)), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
             self._read_csv(os.path.join(data_dir, hp.test_data)), "test")       
        
    def get_labels(self):
        """See base class."""
        return list(hp.dict_id2label.keys())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            #self.labels.add(label)
            # by chenming           
            for l in label:
                self.labels.add(l)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples
    
    

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, task_name):
  """Converts a single `InputExample` into a single `InputFeatures`."""
 
  # by chenming
  task_name =  "sts-b"
  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,#??
        is_real_example=False)

  if task_name != "sts-b":
    label_map = {}
    for (i, label) in enumerate(label_list):
      label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in ALBERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if task_name != "sts-b":
    label_id = label_map[example.label]
  else:
    label_id = example.label

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file, task_name):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer, task_name)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_float_feature([feature.label_id])\
        if task_name == "sts-b" else create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, task_name, use_tpu, bsz,
                                multiple=1):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  labeltype = tf.float32 if task_name == "sts-b" else tf.int64

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "label_ids": tf.FixedLenFeature([], labeltype),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    if use_tpu:
      batch_size = params["batch_size"]
    else:
      batch_size = bsz

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        contrib_data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def _create_model_from_hub(hub_module, is_training, input_ids, input_mask,
                           segment_ids):
  """Creates an ALBERT model from TF-Hub."""
  tags = set()
  if is_training:
    tags.add("train")
  albert_module = hub.Module(hub_module, tags=tags, trainable=True)
  albert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  albert_outputs = albert_module(
      inputs=albert_inputs,
      signature="tokens",
      as_dict=True)
  output_layer = albert_outputs["pooled_output"]
  return output_layer


def _create_model_from_scratch(albert_config, is_training, input_ids,
                               input_mask, segment_ids, use_one_hot_embeddings):
  """Creates an ALBERT model from scratch (as opposed to hub)."""
  model = modeling.AlbertModel(
      config=albert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)
  output_layer = model.get_pooled_output()
  return output_layer


def create_model(albert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, task_name,
                 hub_module):
  """Creates a classification model."""
  if hub_module:
    tf.logging.info("creating model from hub_module: %s", hub_module)
    output_layer = _create_model_from_hub(hub_module, is_training, input_ids,
                                          input_mask, segment_ids)
  else:
    tf.logging.info("creating model from albert_config")
    output_layer = _create_model_from_scratch(albert_config, is_training,
                                              input_ids, input_mask,
                                              segment_ids,
                                              use_one_hot_embeddings)

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    if task_name != "sts-b":
      probabilities = tf.nn.softmax(logits, axis=-1)
      predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    else:
      probabilities = logits
      logits = tf.squeeze(logits, [-1])
      predictions = logits
      per_example_loss = tf.square(logits - labels)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, probabilities, logits, predictions)


def model_fn_builder(albert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, task_name, hub_module=None,
                     optimizer="adamw"):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, probabilities, logits, predictions) = \
        create_model(albert_config, is_training, input_ids, input_mask,
                     segment_ids, label_ids, num_labels,
                     use_one_hot_embeddings, task_name, hub_module)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps,
          use_tpu, optimizer)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      if task_name not in ["sts-b", "cola"]:
        def metric_fn(per_example_loss, label_ids, logits, is_real_example):
          predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
          accuracy = tf.metrics.accuracy(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)
          loss = tf.metrics.mean(
              values=per_example_loss, weights=is_real_example)
          return {
              "eval_accuracy": accuracy,
              "eval_loss": loss,
          }
      elif task_name == "sts-b":
        def metric_fn(per_example_loss, label_ids, logits, is_real_example):
          """Compute Pearson correlations for STS-B."""
          # Display labels and predictions
          concat1 = contrib_metrics.streaming_concat(logits)
          concat2 = contrib_metrics.streaming_concat(label_ids)

          # Compute Pearson correlation
          pearson = contrib_metrics.streaming_pearson_correlation(
              logits, label_ids, weights=is_real_example)

          # Compute MSE
          # mse = tf.metrics.mean(per_example_loss)
          mse = tf.metrics.mean_squared_error(
              label_ids, logits, weights=is_real_example)

          loss = tf.metrics.mean(
              values=per_example_loss,
              weights=is_real_example)

          return {"pred": concat1, "label_ids": concat2, "pearson": pearson,
                  "MSE": mse, "eval_loss": loss,}
      elif task_name == "cola":
        def metric_fn(per_example_loss, label_ids, logits, is_real_example):
          """Compute Matthew's correlations for STS-B."""
          predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
          # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
          tp, tp_op = tf.metrics.true_positives(
              predictions, label_ids, weights=is_real_example)
          tn, tn_op = tf.metrics.true_negatives(
              predictions, label_ids, weights=is_real_example)
          fp, fp_op = tf.metrics.false_positives(
              predictions, label_ids, weights=is_real_example)
          fn, fn_op = tf.metrics.false_negatives(
              predictions, label_ids, weights=is_real_example)

          # Compute Matthew's correlation
          mcc = tf.div_no_nan(
              tp * tn - fp * fn,
              tf.pow((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.5))

          # Compute accuracy
          accuracy = tf.metrics.accuracy(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)

          loss = tf.metrics.mean(
              values=per_example_loss,
              weights=is_real_example)

          return {"matthew_corr": (mcc, tf.group(tp_op, tn_op, fp_op, fn_op)),
                  "eval_accuracy": accuracy, "eval_loss": loss,}

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
              "probabilities": probabilities,
              "predictions": predictions
          },
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, task_name):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  #print('1'*20)
  print('Length of examples:',len(examples))
  for (ex_index, example) in enumerate(examples):
    #print('2'*20)
    #print('ex_index:',ex_index)
    if ex_index % 10000 == 0:
      #print('3'*20) 
      #tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
      print("Writing example %d of %d" % (ex_index, len(examples)))
    #print('4'*20)
    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer, task_name)

    features.append(feature)
  return features


# Load parameters
max_seq_length = hp.sequence_length
do_lower_case = hp.do_lower_case
vocab_file = hp.vocab_file 
tokenizer = tokenization.FullTokenizer.from_scratch(vocab_file=vocab_file,
                                                    do_lower_case=do_lower_case, 
                                                    spm_model_file=None)                               
processor = ClassifyProcessor() 
label_list = processor.get_labels()
data_dir = hp.data_dir 


def get_features():
    # Load train data
    train_examples = processor.get_train_examples(data_dir) 
    # Get onehot feature
    features = convert_examples_to_features( train_examples, label_list, max_seq_length, tokenizer,task_name='classify')
    input_ids = [f.input_ids for f in features]
    input_masks = [f.input_mask for f in features]
    segment_ids = [f.segment_ids for f in features]
    label_ids = [f.label_id for f in features]
    print('Get features finished!')
    return input_ids,input_masks,segment_ids,label_ids

def get_features_test():
    # Load test data
    train_examples = processor.get_test_examples(data_dir) 
    # Get onehot feature
    features = convert_examples_to_features( train_examples, label_list, max_seq_length, tokenizer,task_name='classify_test')
    input_ids = [f.input_ids for f in features]
    input_masks = [f.input_mask for f in features]
    segment_ids = [f.segment_ids for f in features]
    label_ids = [f.label_id for f in features]
    print('Get features(test) finished!')
    return input_ids,input_masks,segment_ids,label_ids


def create_example(line,set_type):
    """Creates examples for the training and dev sets."""
    guid = "%s-%s" % (set_type, 1)
    text_a = tokenization.convert_to_unicode(line[1])
    label = tokenization.convert_to_unicode(line[0])
    example = InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
    return example


def get_feature_test(sentence):
    example = create_example(['0',sentence],'test')
    feature = convert_single_example(0, example, label_list,max_seq_length, tokenizer,task_name='classify')                                    
    return feature.input_ids,feature.input_mask,feature.segment_ids,feature.label_id


if __name__ == '__main__':
    ## Test:获取参数
    sentence = '好好学习，天天向上。'
    feature = get_feature_test(sentence)
    print(feature)
    
