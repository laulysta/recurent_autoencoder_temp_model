import numpy as np
import fcntl  # copy
import itertools
import sys, os
import argparse
import time
import datetime
from temporal_models import train
from os.path import join as pjoin

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-mv', '--model_version', required=False, default='gru', help='ex: gru, lstm, rnn')
parser.add_argument('-dw', '--dim_word', required=False, default='50', help='Size of the word representation')
parser.add_argument('-d', '--dim_model', required=False, default='200', help='Size of the hidden representation')
parser.add_argument('-l', '--lr', required=False, default='0.001', help='learning rate')
parser.add_argument('-data', '--dataset', required=False, default='test', help='ex: test, AP_news')
parser.add_argument('-bs', '--batch_size', required=False, default='64', help='Size of the batch')
parser.add_argument('-rec_c', '--rec_coeff', required=False, default='1.0', help='coefficient for the reconstruction')
parser.add_argument('-out', '--out_dir', required=False, default='.', help='Output directory for the model')


args = parser.parse_args()

model_version = args.model_version
dim_word = int(args.dim_word)
dim_model = int(args.dim_model)
lr = float(args.lr)
dataset = args.dataset
batch_size = int(args.batch_size)
rec_coeff = float(args.rec_coeff)


def makedirs_catchExep(dirPath):
    if not os.path.exists(dirPath):
        try:
            os.makedirs(dirPath)
        except OSError as e:
            print e
            print 'Exeption was catch, will continue script \n'

# Create names and folders
####################################################################################
dirPath = pjoin(args.out_dir, 'saved_temporal_models/')
makedirs_catchExep(dirPath)

if dataset == "test" or dataset == "AP_news" or dataset == "fil9" or dataset == "fil9_small":
    dirModelName = "model_" + "_".join([str(dataset), str(batch_size), str(model_version), str(dim_word), str(dim_model), str(rec_coeff)])
else:
    sys.exit("Wrong dataset")

dirModelPath = pjoin(dirPath, dirModelName)
makedirs_catchExep(dirModelPath)

modelName = os.path.join(dirModelPath, dirModelName + ".npz")

###################################################################################


if dataset == "test":
    n_words = 22
    trainsetPath = '../data/data_test/trainset.txt'
    validsetPath = '../data/data_test/validset.txt'
    testsetPath = '../data/data_test/testset.txt'
elif dataset == "AP_news":
    n_words = 17964
    trainsetPath = '../data/data_AP_news/trainset.txt'
    validsetPath = '../data/data_AP_news/validset.txt'
    testsetPath = '../data/data_AP_news/testset.txt'
elif dataset == "fil9":
    n_words = 20652
    trainsetPath = '../data/data_fil9/trainset.txt'
    validsetPath = '../data/data_fil9/validset.txt'
    testsetPath = '../data/data_fil9/testset.txt'
elif dataset == "fil9_small":
    n_words = 20652
    trainsetPath = '../data/data_fil9_small/trainset.txt'
    validsetPath = '../data/data_fil9_small/validset.txt'
    testsetPath = '../data/data_fil9_small/testset.txt'

reload_ = False


trainerr, validerr, testerr = train(saveto=modelName,
                                    reload_=reload_,
                                    dim_word=dim_word,
                                    dim=dim_model,
                                    model_version=model_version,
                                    max_epochs=100,
                                    n_words=n_words,
                                    optimizer='adadelta',
                                    decay_c=0.,
                                    diag_c=0.,  # not used with adadelta
                                    lrate=lr,
                                    patience=5,
                                    batch_size=batch_size,
                                    valid_batch_size=batch_size,
                                    trainsetPath=trainsetPath,
                                    validsetPath=validsetPath,
                                    testsetPath=testsetPath,
                                    clip_c=1.,
                                    rec_coeff=rec_coeff)


# Prepare result line to append to result file
line = "\t".join([str(dirModelName), str(dataset), str(batch_size), str(model_version), str(dim_word), str(dim_model), str(rec_coeff), str(np.exp(trainerr)), str(np.exp(validerr)), str(np.exp(testerr))]) + "\n"

# Preparing result file
results_file = dirPath + 'results.txt'
if not os.path.exists(results_file):
    # Create result file if doesn't exist
    header_line = "\t".join(['dirModelName', 'dataset', 'batch_size', 'model_version', 'dim_word', 'dim_model', 'rec_coeff',
                             'train_perplexity', 'valid_perplexity', 'test_perplexity']) + '\n'
    f = open(results_file, 'w')
    f.write(header_line)
    f.close()

f = open(results_file, "a")
fcntl.flock(f.fileno(), fcntl.LOCK_EX)
f.write(line)
f.close()  # unlocks the file
