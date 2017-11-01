from keras.models import load_model
import argparse
from keras.layers import SimpleRNN, GRU, LSTM
import pickle
import re
import numpy as np
from processing_arithmetics.sequential.architectures import DiagnosticClassifier, ScalarPrediction
from processing_arithmetics.arithmetics.treebanks import treebank
from argument_transformation import max_length, get_hidden_layer

"""
Train a diagnostic classifier for an existing model.
"""

###################################################
# Set some params
digits = np.arange(-10, 11)
operators = ['+', '-']

###################################################
# Create argument parser

parser = argparse.ArgumentParser()
parser.add_argument("-classifiers", required=True, nargs="*", choices=['subtracting', 'intermediate_locally', 'intermediate_recursively', 'grammatical', 'intermediate_directly', 'depth', 'minus1depth', 'minus2depth', 'minus3depth', 'minus4depth', 'minus1depth_count', 'switch_mode'])
parser.add_argument("--nb_epochs", type=int, required=True)
parser.add_argument("--save_to", help="Save model to filename")
parser.add_argument("--hidden", type=get_hidden_layer, help="Hidden layer type", choices=[SimpleRNN, GRU, LSTM])

parser.add_argument("--seed", type=int, help="Set random seed", default=8)
parser.add_argument("-N", type=int, help="Run script N times", default=1)
parser.add_argument("--format", type=str, help="Set formatting of arithmetic expressions", choices=['infix', 'postfix', 'prefix'], default="infix")
parser.add_argument("--seed_test", type=int, help="Set random seed for testset", default=100)
parser.add_argument("--test_gates", action="store_true", help="Run diagnostic classifier on gates instead of hidden layer activations")

parser.add_argument("--optimizer", help="Set optimizer for training", choices=['adam', 'adagrad', 'adamax', 'adadelta', 'rmsprop', 'sgd'], default='adam')
parser.add_argument("--dropout", help="Set dropout fraction", default=0.0)
parser.add_argument("-b", "--batch_size", help="Set batch size", default=24)
parser.add_argument("--val_split", help="Set validation split", default=0.1)

parser.add_argument("-maxlen", help="Set maximum number of digits in expression that network should be able to parse", type=max_length, default=max_length(15))
parser.add_argument("--verbosity", type=int, choices=[0, 1, 2], default=2)
parser.add_argument("--debug", action="store_true", help="Run with small treebank for debugging")
parser.add_argument("--target_folder", help="Set folder to store models", default="")

args = parser.parse_args()

####################################################
# Set some params
languages_train             = treebank(seed=args.seed, kind='train', debug=args.debug)
languages_val              = treebank(seed=args.seed, kind='heldout', debug=args.debug)
languages_test              = [(name, tb) for name, tb in treebank(seed=args.seed_test, kind='test', debug=args.debug)]

results_all = {}

training_data = None
validation_data = None

eval_filename = args.save_to+'_evaluation'
results_name = args.save_to+'.results'
eval_file = open(eval_filename, 'w')
results_all = dict()


model = ScalarPrediction(digits=digits, operators=operators)

for n in xrange(args.N):

    model.generate_model(args.hidden, input_size=2,
        input_length=args.maxlen, size_hidden=15)
    
    print("\nTraining diagnostic classifier %i " % n)
    save_to = args.save_to+str(n)

    training = DiagnosticClassifier(digits=digits, operators=operators, model=model.model, classifiers=args.classifiers)

    training_data = training_data or training.generate_training_data(languages_train, format=args.format)
    validation_data = validation_data or training.generate_training_data(languages_val, format=args.format)

    training.train(training_data=training_data, validation_data=validation_data, 
            validation_split=args.val_split, batch_size=args.batch_size,
            epochs=args.nb_epochs, verbosity=args.verbosity, filename=save_to+'n',
            save_every=False)


    ######################################################################################
    # Test model and write to file

    # generate_test_data
    test_data = training.generate_test_data(data=languages_test, digits=digits, format=args.format)

    evaluation = training.test(test_data)
    eval_str = training.evaluation_string(evaluation)

    results_all[n] = evaluation


# dump all results
pickle.dump(results_all, open(results_name,'wb'))

