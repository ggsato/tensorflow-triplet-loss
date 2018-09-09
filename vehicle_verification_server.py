#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import os
import json
import traceback

import tornado
import tornado.ioloop
import tornado.web

from io import BytesIO
import cv2
import numpy as np

import logging

import argparse
import os

import tensorflow as tf

from model.input_fn_trackings import predict_input_fn
from model.model_fn import model_fn
from model.utils import Params

__UPLOADS__ = "/dev/shm/vefifer/uploads/"
if not os.path.exists(__UPLOADS__):
    os.makedirs(__UPLOADS__)

class VehicleVerifer():
    def __init__(self, model_dir, id_threshold, params):

        # Define the model
        tf.logging.info("Restoring the model...")
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=model_dir)
        self._id_threshold = id_threshold
        self._params = params
        self._estimator = tf.estimator.Estimator(model_fn, params=self._params, config=config)

    def verify(self, image0, image1):
        
        predictions = self._estimator.predict(lambda: predict_input_fn(image0, image1, self._params))

        distance = next(predictions)['from_distances'][1]
        print('distance between {} and {} = {}'.format(image0, image1, distance))

        return distance < self._id_threshold

class Errors(tornado.web.RequestHandler):
    def get(self):
        global errors
        self.finish(str(errors))

class Userform(tornado.web.RequestHandler):
    def get(self):
        self.render("upload.html")


class Upload(tornado.web.RequestHandler):
    def post(self):
        global errors, vehicle_verifer
        message = ''

        tmp0 = None
        tmp1 = None

        try:

            images = self.request.files['images']

            if len(images) != 2:
                print('the number of images should be two for verification')

            else:

                fileinfo0 = images[0]
                fileinfo1 = images[1]
                
                print('type of fileinfo body = {}'.format(type(fileinfo0['body'])))

                img_byte0 = BytesIO(fileinfo0['body'])
                image0 = cv2.imdecode(np.frombuffer(img_byte0.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

                tmp0 = os.path.join(__UPLOADS__, 'image0.jpg')
                cv2.imwrite(tmp0, image0)

                img_byte1 = BytesIO(fileinfo1['body'])
                image1 = cv2.imdecode(np.frombuffer(img_byte1.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
                tmp1 = os.path.join(__UPLOADS__, 'image1.jpg')
                cv2.imwrite(tmp1, image1)

                # verify
                verified = vehicle_verifer.verify(tmp0, tmp1)

                message = str(verified)

        except:
            info = sys.exc_info()
            logging.exception('Unknow Exception {}, {}, {}'.format(info[0], info[1], info[2]))
            traceback.print_tb(info[2])
            errors += 1
        finally:

            if tmp0 is not None:
                os.remove(tmp0)

            if tmp1 is not None:
                os.remove(tmp1)

        self.finish(message)

application = tornado.web.Application([
        (r"/", Userform),
        (r"/upload", Upload),
        (r"/errors", Errors),
        ], debug=True)

errors = 0
vehicle_verifer = None

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

    vehicle_verifer = VehicleVerifer(args.model_dir, id_threshold, params)

    # start loop
    print('starting tornado loop...')
    tornado.ioloop.IOLoop.instance().start()