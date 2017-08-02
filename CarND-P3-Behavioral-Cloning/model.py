from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam
from keras import callbacks

import argparse, pickle

from matplotlib import pyplot as plt

from utils import INPUT_SHAPE, load_data, split_data, train_generator, valid_generator

def build_nvidia(input_shape, kinit='glorot_uniform', activation='elu', use_dropout = True, keep_prob=0.5):
    model = Sequential()

    # simple normalization
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape, name='normalize'))

    # first 3 conv-layers use kernel (5, 5) with stride (2, 2)
    # filters-depths are [24, 36, 48]
    filter_depths = [24, 36, 48]
    layer = 1
    for d in filter_depths:
        model.add(Conv2D(d, 5, strides=(2, 2), activation=activation, padding='valid',
                         name='conv{}'.format(layer), kernel_initializer=kinit))
        layer += 1

    # the last 2 conv-layers use kernel (3, 3) with stride (1, 1)
    filter_depths = [64, 64]
    for d in filter_depths:
        model.add(Conv2D(d, 3, strides=(1, 1), activation=activation, padding='valid',
                         name='conv{}'.format(layer), kernel_initializer=kinit))
        layer += 1

    # flatten
    model.add(Flatten(name='flatten'))
    if use_dropout:
        model.add(Dropout(keep_prob, name='dropout_conv'))

    # fully-connected layers
    hidden_dims = [100, 50, 10]
    layer = 1
    for h in hidden_dims:
        model.add(Dense(h, activation=activation, kernel_initializer=kinit,
                        name='hidden{}'.format(layer)))

        # model.add(Dropout(0.5, name='dropout_h{}'.format(layer)))

        layer += 1

    # output-regressor
    model.add(Dense(1, name='output', kernel_initializer=kinit))

    return model


def train_model(model,
                train_data_gen, steps_per_epoch,
                valid_data_gen, validation_steps,
                epochs,
                learning_rate=1e-4,
                save_file='model_last.h5',
                best_file='model_best.h5',
                history_file='history.pkl'):

    optimizer = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)

    # save model has best valuation
    save_best = callbacks.ModelCheckpoint(best_file, save_best_only=True)

    # train-loop
    history_object = model.fit_generator(train_data_gen, steps_per_epoch, epochs,
                                         validation_data=valid_data_gen,
                                         validation_steps=validation_steps,
                                         callbacks=[save_best])

    # save trained model
    model.save(save_file)

    # save history
    with open(history_file, 'wb') as f:
        pickle.dump(history_object.history, f)

    # print some logging
    print('\n-------------------------------------')
    print('Training is done, model is saved to')
    print('\tlast epoch      {}'.format(save_file))
    print('\tbest validation {}'.format(best_file))
    print('Training history is saved to {}'.format(history_file))

    return history_object.history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument('-a', help='activation type', dest='activation', type=str, default='elu')
    parser.add_argument('--kernel_init', help='weight initializer',     dest='kernel_init', type=str,   default='glorot_uniform')
    parser.add_argument('--use_dropout', help='add dropout after conv', dest='use_dropout', action='store_true')
    parser.add_argument('--keep_prob',   help='keep probability',       dest='keep_prob',   type=float, default=0.5)

    # data parameters
    parser.add_argument('-d', help='data directory',                   dest='data_dir',   type=str, default='data')
    parser.add_argument('--split_frac',   help='split train/validation', dest='split_frac', type=float, default=0.8)
    parser.add_argument('--steer_corr',   help='steering correction',    dest='steer_corr', type=float, default=0.2)

    # training parameters
    parser.add_argument('-n', help='number of epochs', dest='nb_epochs',  type=int, default=20)
    parser.add_argument('-b', help='batch size',       dest='batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', help='learning rate', dest='learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_file',     help='output file after last epoch',   dest='save_file', type=str, default='model_last.h5')
    parser.add_argument('--best_file',     help='output file with best val-loss', dest='best_file', type=str, default='model_best.h5')

    # save history file
    parser.add_argument('--hist_file', help='output file with best val-loss', dest='hist_file', type=str,
                        default='history.pkl')

    args = parser.parse_args()

    # build model
    model = build_nvidia(INPUT_SHAPE,
                         kinit=args.kernel_init,
                         activation=args.activation,
                         use_dropout=args.use_dropout,
                         keep_prob=args.keep_prob)

    # print model summary
    model.summary()

    # get data
    dataset = load_data(args.data_dir)

    # split to train/validation
    train, valid = split_data(dataset, args.split_frac)

    # train/valid generator: we always use multiple camera for training
    train_gen = train_generator(train, INPUT_SHAPE, args.batch_size, args.steer_corr, True)
    valid_gen = valid_generator(valid, INPUT_SHAPE, args.batch_size)

    steps_per_epoch  = train.shape[0] // args.batch_size
    validation_steps = valid.shape[0] // args.batch_size

    print('\n-------------------------------------------------')
    print('Start training with following hyperparameters')
    print('\tnumber of epochs {}'.format(args.nb_epochs))
    print('\tlearning rate    {:.2e}'.format(args.learning_rate))
    print('-------------------------------------------------\n')

    # train model
    history = train_model(model,
                         train_gen, steps_per_epoch,
                         valid_gen, validation_steps,
                         args.nb_epochs,
                         learning_rate=args.learning_rate,
                         save_file=args.save_file,
                         best_file=args.best_file,
                         history_file=args.hist_file)


    ### plot the training and validation loss for each epoch
    # plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    # plt.title('model mean squared error loss')
    # plt.ylabel('mean squared error loss')
    # plt.xlabel('epoch')
    # plt.legend(['training set', 'validation set'], loc='upper right')
    # plt.show()

