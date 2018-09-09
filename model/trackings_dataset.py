import os, math
from datetime import datetime

import tensorflow as tf
import numpy as np

is_color = False

def train(directory, train_size, params):
    """tf.data.Dataset object for tracking training data."""
    return dataset(directory, train_size, params)


def test(directory, eval_size, params):
    """tf.data.Dataset object for tracking test data."""
    return dataset(directory, eval_size, params)

def predict(images, params):
    """tf.data.Dataset object for tracking predict data"""
    return prediction_dataset(images, params)

def dataset(directory, train_or_eval_size, params):
    """ create a dataset from a directory """

    global is_color
    if params.is_color != is_color:
        is_color = params.is_color

    filenames, labels = read_image_files_and_labels(directory, train_or_eval_size)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_read_image_from_file)

    return dataset

def read_image_files_and_labels(directory, train_or_eval_size, as_tf_constatns=True):
    """ traverse a directory and returns a set of files and their labels

    Args:
        directory: ROOT folder path consisting of ROOT/LOCATION/YEAR/MONTH/DAY/trajectories/TRACKING_ID/*.jpg

    Returns:
        filenames: a list of filenames as tf constants
        labels: a list of labels as tf constants

    """

    filenames = []
    labels = []

    locations = sorted(os.listdir(directory))
    done = False
    tracking_images = []
    for location in locations:
        location_path = os.path.join(directory, location)
        if not os.path.isdir(location_path):
            continue
        years = sorted(os.listdir(location_path))
        for year in years:
            year_path = os.path.join(location_path, year)
            if not os.path.isdir(year_path):
                continue
            months = sorted(os.listdir(year_path))
            for month in months:
                month_path = os.path.join(year_path, month)
                if not os.path.isdir(month_path):
                    continue
                days = sorted(os.listdir(month_path))
                for day in days:
                    day_path = os.path.join(month_path, day)
                    if not os.path.isdir(day_path):
                        continue

                    trajectory_path = os.path.join(day_path, 'trajectories')

                    #print('listing trackings in {} ...'.format(trajectory_path))
                    tracking_ids = os.listdir(trajectory_path)

                    #print('sorting trackings ...')
                    tracking_ids = sorted(tracking_ids)

                    for tracking_id in tracking_ids:
                        tracking_id_path = os.path.join(trajectory_path, tracking_id)
                        if not os.path.isdir(tracking_id_path):
                            continue

                        tracking_image = 0
                        files = sorted(os.listdir(tracking_id_path))
                        for file in files:
                            if not file.endswith('.jpg'):
                                continue

                            # tracking_id as label
                            label = tracking_id_as_number(tracking_id)
                            filename = os.path.join(tracking_id_path, file)

                            #print('adding {} of tracking_id:{} as {}'.format(filename, tracking_id, label))

                            filenames.append(filename)
                            labels.append(label)
                            tracking_image += 1

                            if len(filenames) >= train_or_eval_size:
                                tracking_images.append(tracking_image)
                                print('average tracking images = {:3.1f}'.format(sum(tracking_images)/len(tracking_images)))
                                if as_tf_constatns:
                                    return tf.constant(filenames), tf.constant(labels)
                                else:
                                    return filenames, labels

                        tracking_images.append(tracking_image)

    print('average tracking images = {:3.1f}'.format(sum(tracking_images)/len(tracking_images)))


    if as_tf_constatns:
        return tf.constant(filenames), tf.constant(labels)

    return filenames, labels

EPOCH_DATETIME = datetime(1970, 1, 1)
def tracking_id_as_number(tracking_id):

    tracking_number_index = tracking_id.index('-')
    tracking_timestamp = datetime.strptime(tracking_id[:tracking_number_index], "%Y%m%d_%H%M%S.%f")
    epoch_seconds = int(math.floor((tracking_timestamp - EPOCH_DATETIME).total_seconds()))
    tracking_number = int(tracking_id[tracking_number_index+1:])
    tracking_id_number = epoch_seconds * 1000 * 1000 + tracking_timestamp.microsecond + tracking_number
    #print('tracking_timestamp {}: {} + {} + {} => {}'.format(tracking_timestamp, epoch_seconds, tracking_timestamp.microsecond, tracking_number, tracking_id_number))

    return tracking_id_number

def _read_image_from_file(filename, label):
    """ returns a single sample with the given label by reading an image from a file

    Args:
        filename: a path of a sample image
        label: a label of a sample

    Returns:
        image_resized: a resized image array
        label: a label of a sample

    """

    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=0 if is_color else 1)
    image_scaled = tf.divide(image_decoded, tf.constant(255, dtype=tf.uint8))
    image_resized = tf.image.resize_images(image_scaled, [28, 28])

    return image_resized, label

def prediction_dataset(images, params):
    """ create a dataset from a directory """

    global is_color
    if params.is_color != is_color:
        is_color = params.is_color

    filenames = tf.constant(images)
    labels = tf.constant(np.ones((len(images))))

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_read_image_from_file)

    return dataset