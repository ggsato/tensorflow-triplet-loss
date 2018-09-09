"""Train the model"""

import argparse
import os

import tensorflow as tf

from model.input_fn_trackings import predict_input_fn
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

    # Train the model
    images = [args.image0, args.image1]
    predictions = estimator.predict(lambda: predict_input_fn(images, params))

    print('predictions = {}'.format(predictions))
    for pred_dict in predictions:
        for key in pred_dict:
            print('{} = {}'.format(key, pred_dict[key]))