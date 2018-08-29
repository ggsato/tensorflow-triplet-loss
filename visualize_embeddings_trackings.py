"""Train the model"""

import argparse
import os
import pathlib
import shutil
import math

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import cv2

import model.trackings_dataset as trackings_dataset
from model.trackings_dataset import read_image_files_and_labels
from model.utils import Params
from model.input_fn_trackings import test_input_fn
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/tracking_model',
                    help="Experiment directory containing params_trackings.json")
parser.add_argument('--sprite_filename', default='experiments/trackings_sprite.png',
                    help="Sprite image for the projector")

def generate_sprite_image(sprite_filename, data_dir, eval_size, image_size):
    print('generating a sprite image as {}'.format(sprite_filename))
    filenames, _ = read_image_files_and_labels(data_dir, eval_size, as_tf_constatns=False)
    dim_count = int(math.ceil(math.sqrt(eval_size)))
    sprite_array = np.zeros([dim_count * image_size, dim_count * image_size, 3])
    row = 0
    col = 0
    for filename in filenames:
        #img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(filename)
        fx = image_size / img.shape[1]
        fy = image_size / img.shape[0]
        resized = cv2.resize(img, None, fx=fx, fy=fy)
        sprite_array[row * image_size:(row + 1) * image_size, col * image_size:(col + 1) * image_size, :] = resized

        col += 1
        if col == dim_count:
            col = 0
            row += 1

    cv2.imwrite(sprite_filename, sprite_array)

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params_trackings.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)


    # EMBEDDINGS VISUALIZATION

    # Compute embeddings on the test set
    tf.logging.info("Predicting")
    predictions = estimator.predict(lambda: test_input_fn(args.data_dir, params))

    embeddings = np.zeros((params.eval_size, params.embedding_size))
    for i, p in enumerate(predictions):
        embeddings[i] = p['embeddings']

    tf.logging.info("Embeddings shape: {}".format(embeddings.shape))

    # Visualize test embeddings
    embedding_var = tf.Variable(embeddings, name='trackings_embedding')

    eval_dir = os.path.join(args.model_dir, "eval")
    summary_writer = tf.summary.FileWriter(eval_dir)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # generate sprite image
    generate_sprite_image(args.sprite_filename, args.data_dir, params.eval_size, params.image_size)

    # Specify where you find the sprite (we will create this later)
    # Copy the embedding sprite image to the eval directory
    shutil.copy2(args.sprite_filename, eval_dir)
    embedding.sprite.image_path = pathlib.Path(args.sprite_filename).name
    embedding.sprite.single_image_dim.extend([28, 28])

    with tf.Session() as sess:
        # Obtain the test labels
        dataset = trackings_dataset.test(args.data_dir, params.eval_size, params)
        dataset = dataset.map(lambda img, lab: lab)
        dataset = dataset.batch(params.eval_size)
        labels_tensor = dataset.make_one_shot_iterator().get_next()
        labels = sess.run(labels_tensor)

    # Specify where you find the metadata
    # Save the metadata file needed for Tensorboard projector
    metadata_filename = "vehicels_metadata.tsv"
    with open(os.path.join(eval_dir, metadata_filename), 'w') as f:
        for i in range(params.eval_size):
            c = labels[i]
            f.write('{}\n'.format(c))
    embedding.metadata_path = metadata_filename

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(embedding_var.initializer)
        saver.save(sess, os.path.join(eval_dir, "embeddings.ckpt"))
