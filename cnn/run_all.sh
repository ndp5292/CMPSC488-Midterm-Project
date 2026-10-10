#!/bin/bash
# Runs every CNN experiment on the team's fixed splits (data/train.csv, validation.csv, test.csv)
D=${1:-data}
A="--train $D/train.csv --val $D/validation.csv --test $D/test.csv --seeds 0 1 2"
for m in output input both; do python cnn_classifier.py $A --mode $m; done
for ab in format punct length; do python cnn_classifier.py $A --mode output --ablate $ab; done
