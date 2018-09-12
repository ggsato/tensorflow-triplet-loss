"""Train the model"""

import argparse
import os

import tensorflow as tf

from model.trackings_dataset import predictor_dataset
from model.model_fn import model_fn
from model.utils import Params

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/tracking_model',
                    help="Experiment directory containing params_trackings.json")
parser.add_argument('image0', help="a path to image0 to compute a distance to image1")
parser.add_argument('image1', help="another path to image1 to compute a distance from image0")

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params_trackings.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Define the model
    tf.logging.info("Restoring the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # predict with the model
    def serving_input_receiver_fn():
        features = {'images': tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 3], name='images')}
        return tf.estimator.export.ServingInputReceiver(features=features, receiver_tensors=features)

    predictor = tf.contrib.predictor.from_estimator(estimator, serving_input_receiver_fn) 

    images = predictor_dataset([args.image0, args.image1], params)

    predictions = predictor({'images': images})

    embeddings = predictions['embeddings']
    from_distances = predictions['from_distances']

    print('from_distances = {}'.format(from_distances))