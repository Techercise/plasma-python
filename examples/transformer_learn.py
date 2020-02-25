from plasma.models.loader import Loader
from plasma.preprocessor.preprocess import guarantee_preprocessed
from plasma.transformer.runner import train
from plasma.models.torch_runner import make_predictions_and_evaluate_gpu
from plasma.conf import conf

from pprint import pprint
import numpy as np
import datetime
import logging
import random
import sys
import os

import matplotlib
matplotlib.use('Agg')

pprint(conf)

if conf['data']['normalizer'] == 'minmax':
    from plasma.preprocessor.normalize import MinMaxNormalizer as Normalizer
elif conf['data']['normalizer'] == 'meanvar':
    from plasma.preprocessor.normalize import MeanVarNormalizer as Normalizer
elif conf['data']['normalizer'] == 'var':
    # performs !much better than minmaxnormalizer
    from plasma.preprocessor.normalize import VarNormalizer as Normalizer
elif conf['data']['normalizer'] == 'averagevar':
    # performs !much better than minmaxnormalizer
    from plasma.preprocessor.normalize import (
        AveragingVarNormalizer as Normalizer
    )
else:
    print('unkown normalizer. exiting')
    exit(1)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s",
    )
    LOGGER = logging.getLogger("transformer_learn")

    shot_list_dir = conf['paths']['shot_list_dir']
    shot_files = conf['paths']['shot_files']
    shot_files_test = conf['paths']['shot_files_test']
    train_frac = conf['training']['train_frac']
    stateful = conf['model']['stateful']

    # FIXME change seed setting
    np.random.seed(0)
    random.seed(0)

    only_predict = len(sys.argv) > 1
    custom_path = None
    if only_predict:
        custom_path = sys.argv[1]
        print("predicting using path {}".format(custom_path))

    #####################################################
    #                   PREPROCESSING                   #
    #####################################################
    # TODO(KGF): check tuple unpack
    (shot_list_train, shot_list_validate,
     shot_list_test) = guarantee_preprocessed(conf)

    #####################################################
    #                   NORMALIZATION                   #
    #####################################################

    print("normalization", end='')
    nn = Normalizer(conf)
    nn.train()
    loader = Loader(conf, nn)
    print("...done")
    print('Training on {} shots, testing on {} shots'.format(
        len(shot_list_train), len(shot_list_test)))

    #####################################################
    #                    TRAINING                       #
    #####################################################
    train(conf, shot_list_train.random_sublist(512),
          shot_list_validate.random_sublist(256), loader)
    # if not only_predict:
    #    p = old_mp.Process(target=train,
    #                       args=(conf, shot_list_train,
    #                             shot_list_validate, loader)
    #                       )
    #    p.start()
    #    p.join()

    #####################################################
    #                    PREDICTING                     #
    #####################################################
    loader.set_inference_mode(True)

    # load last model for testing
    print('saving results')
    y_prime = []
    y_prime_test = []
    y_prime_train = []

    y_gold = []
    y_gold_test = []
    y_gold_train = []

    disruptive = []
    disruptive_train = []
    disruptive_test = []

    # y_prime_train, y_gold_train, disruptive_train =
    #         make_predictions(conf, shot_list_train, loader)
    # y_prime_test, y_gold_test, disruptive_test =
    #         make_predictions(conf, shot_list_test, loader)

    # TODO(KGF): check tuple unpack
    (y_prime_train, y_gold_train, disruptive_train, roc_train,
     loss_train) = make_predictions_and_evaluate_gpu(
         conf, shot_list_train, loader, custom_path)
    (y_prime_test, y_gold_test, disruptive_test, roc_test,
     loss_test) = make_predictions_and_evaluate_gpu(
         conf, shot_list_test, loader, custom_path)
    print('=========Summary========')
    print('Train Loss: {:.3e}'.format(loss_train))
    print('Train ROC: {:.4f}'.format(roc_train))
    print('Test Loss: {:.3e}'.format(loss_test))
    print('Test ROC: {:.4f}'.format(roc_test))

    disruptive_train = np.array(disruptive_train)
    disruptive_test = np.array(disruptive_test)

    y_gold = y_gold_train + y_gold_test
    y_prime = y_prime_train + y_prime_test
    disruptive = np.concatenate((disruptive_train, disruptive_test))

    shot_list_validate.make_light()
    shot_list_test.make_light()
    shot_list_train.make_light()

    save_str = 'results_' + datetime.datetime.now().strftime(
        "%Y-%m-%d-%H-%M-%S")
    result_base_path = conf['paths']['results_prepath']
    if not os.path.exists(result_base_path):
        os.makedirs(result_base_path)
    np.savez(result_base_path+save_str, y_gold=y_gold,
             y_gold_train=y_gold_train,
             y_gold_test=y_gold_test,
             y_prime=y_prime, y_prime_train=y_prime_train,
             y_prime_test=y_prime_test, disruptive=disruptive,
             disruptive_train=disruptive_train,
             disruptive_test=disruptive_test,
             shot_list_validate=shot_list_validate,
             shot_list_train=shot_list_train, shot_list_test=shot_list_test,
             conf=conf)

    print('finished.')