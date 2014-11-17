#!/bin/bash

ROOT=$(cd $(dirname $0) && pwd)/..

TMPDIR=/tmp/train_ngram.$$

if [ $# -eq 3 ]; then
    WORKDIR=$3
elif [ $# -eq 2 ]; then
    WORKDIR=$TMPDIR
else
    echo "usage: $0 <infile> <outfile> [<tmpdir>]"
    exit 1
fi

INFILE=$1
OUTFILE=$2
PREFIX=$(basename $OUTFILE)

EPOCHS=10
VOCAB_SIZE=5000
NGRAM_SIZE=3

mkdir -p $WORKDIR

$ROOT/src/prepareNeuralLM --train_text $INFILE --ngram_size $NGRAM_SIZE --vocab_size $VOCAB_SIZE --validation_size 500 --write_words_file $WORKDIR/words --train_file $WORKDIR/train.ngrams --validation_file $WORKDIR/validation.ngrams --mmap_file 1 || exit 1

$ROOT/src/trainNeuralNetwork --train_file $WORKDIR/train.ngrams --validation_file $WORKDIR/validation.ngrams --num_epochs $EPOCHS --words_file $WORKDIR/words --model_prefix $WORKDIR/$PREFIX --learning_rate 1 --minibatch_size 8 --mmap_file  1 --ngram_size $NGRAM_SIZE  || exit 1

cp $WORKDIR/$PREFIX.$(($EPOCHS)) $OUTFILE || exit 1

rm -rf $TMPDIR
