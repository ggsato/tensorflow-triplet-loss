# -*- coding: utf-8 -*-

import sys
import time
import os
import json
import traceback
import shutil

import tornado
import tornado.ioloop
import tornado.web

from io import BytesIO
import cv2
import numpy as np

import logging

import argparse

import tensorflow as tf

from model.input_fn_trackings import predict_input_fn
from model.model_fn import model_fn
from model.utils import Params

__UPLOADS__ = "/dev/shm/vefifer/uploads/"
if not os.path.exists(__UPLOADS__):
    os.makedirs(__UPLOADS__)

class VehicleVerifier():
    def __init__(self, model_dir, id_threshold, params):

        # Define the model
        tf.logging.info("Restoring the model...")
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=model_dir)
        self._id_threshold = id_threshold
        self._params = params
        self._estimator = tf.estimator.Estimator(model_fn, params=self._params, config=config)

        self._references = None

    def verify(self, images):

        # predictions is a generator
        # predict is executed at the first next(predictions)
        start = time.time()
        predictions = self._estimator.predict(lambda: predict_input_fn(images, self._params))
        print('estimator predicted in {}ms'.format(time.time() - start))

        # TODO: this is an optimal assignment problem, so resolve by a Hungarian algorithm if necessary
        identities = []
        for p, prediction in enumerate(predictions):
            # dict
            #print('prediction type = {}'.format(type(prediction)))
            # numpy.ndarray
            from_distances = prediction['from_distances']
            #print('from_distances = {}, shape={}, type = {}'.format(from_distances, from_distances.shape, type(from_distances)))
            # numpy.ndarray
            embeddings = prediction['embeddings']
            #print('embeddings shape = {}, type = {}'.format(embeddings.shape, type(embeddings)))
            above_zero = from_distances > 0
            below_threshold = from_distances < self._id_threshold
            within_threshold = above_zero & below_threshold
            #print('within_threshold = {}'.format(within_threshold))
            locations = np.where(within_threshold)
            #print('locations = {}'.format(locations))
            if locations[0].size == 0:
                print('{} has no identical object'.format(p))
                identities.append(-1)
                continue
            min_location_in_locations = np.argmin(from_distances[locations])
            #print('min_location_in_locations = {}'.format(min_location_in_locations))
            min_location = locations[0][min_location_in_locations]
            #print('min_location = {}'.format(min_location))
            if min_location == p:
                raise Exception('this is itself! this is not supposed to happen by assuming self similarity is zero')
            print('{} has an identical object {} at {}'.format(p, from_distances[min_location], min_location))

            # for debug
            debug = False
            if debug:
                folder = os.path.join(__UPLOADS__, '{}-{}_{}'.format(p, min_location, from_distances[min_location]))
                os.makedirs(folder)
                img0_path = os.path.join(folder, '{}.jpg'.format(p))
                shutil.copyfile(images[p], img0_path)
                img1_path = os.path.join(folder, '{}-ref.jpg'.format(min_location))
                shutil.copyfile(images[min_location], img1_path)

            identities.append(int(min_location))

        print('identities verified in {}ms'.format(time.time() - start))

        return identities

class Errors(tornado.web.RequestHandler):
    def get(self):
        global errors
        self.finish(str(errors))

class Userform(tornado.web.RequestHandler):
    def get(self):
        self.render("upload.html")


class Upload(tornado.web.RequestHandler):
    def post(self):
        global errors, vehicle_verifier
        message = ''

        images = []

        try:

            files = self.request.files['images']

            if len(files) < 2:
                print('the number of images should be more than two for verification')

            else:

                for i, fileinfo in enumerate(files):
                    
                    print('reading {}'.format(fileinfo['filename']))

                    img_byte = BytesIO(fileinfo['body'])
                    image = cv2.imdecode(np.frombuffer(img_byte.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

                    tmp = os.path.join(__UPLOADS__, 'image{}.jpg'.format(i))
                    cv2.imwrite(tmp, image)

                    images.append(tmp)

                # verify
                identities = vehicle_verifier.verify(images)

                message = json.dumps(identities)

        except:
            info = sys.exc_info()
            logging.exception('Unknow Exception {}, {}, {}'.format(info[0], info[1], info[2]))
            traceback.print_tb(info[2])
            errors += 1
        finally:

            for tmp in images:
                os.remove(tmp)

        self.finish(message)

application = tornado.web.Application([
        (r"/", Userform),
        (r"/upload", Upload),
        (r"/errors", Errors),
        ], debug=True)

errors = 0
vehicle_verifier = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', default='experiments/tracking_model',
                        help="Experiment directory containing params_trackings.json")
    parser.add_argument('--port', default='9000',
                        help="the listening port for the verification server")
    parser.add_argument('--id_threshold', default='1.0',
                        help="the threshold to identify")
    args = parser.parse_args()

    # listen
    port = int(args.port)
    print('listening on {}'.format(port))
    application.listen(port)

    # tensorflow
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params_trackings.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    id_threshold = float(args.id_threshold)

    vehicle_verifier = VehicleVerifier(args.model_dir, id_threshold, params)

    # start loop
    print('starting tornado loop...')
    tornado.ioloop.IOLoop.instance().start()