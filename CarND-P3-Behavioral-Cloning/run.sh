#!/bin/bash
set -e

EPOCHS=100

# Experiment 1
python model.py -n $EPOCHS --save_file exp1_last.h5 --best_file exp1_best.h5 --hist_file exp1_hist.pkl > run1.log 2>&1 

# Experiment 2: add Dropout
python model.py -n $EPOCHS --save_file exp2_last.h5 --best_file exp2_best.h5 --hist_file exp2_hist.pkl --use_dropout > run2.log 2>&1

# Experiment 3/4: try different steering correction
python model.py -n $EPOCHS --save_file exp3_last.h5 --best_file exp3_best.h5 --hist_file exp3_hist.pkl --use_dropout --steer_corr 0.15 > run3.log 2>&1
python model.py -n $EPOCHS --save_file exp4_last.h5 --best_file exp4_best.h5 --hist_file exp4_hist.pkl --use_dropout --steer_corr 0.25 > run4.log 2>&1

