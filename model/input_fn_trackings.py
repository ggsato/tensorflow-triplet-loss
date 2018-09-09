"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf

import model.trackings_dataset as trackings_dataset


def train_input_fn(data_dir, params):
    """Train input function for the trackings dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = trackings_dataset.train(data_dir, params.train_size, params)
    #dataset = dataset.shuffle(params.train_size)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset


def test_input_fn(data_dir, params):
    """Test input function for the trackings dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = trackings_dataset.test(data_dir, params.eval_size, params)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset

def predict_input_fn(images, params):
    """ Predict input function to compute a distance between image0 and image1
    """
    dataset = trackings_dataset.predict(images, params)
    dataset = dataset.batch(len(images))
    dataset = dataset.prefetch(1)
    return dataset