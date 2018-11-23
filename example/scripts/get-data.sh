#!/bin/bash

## Using a few examples from the svm pdf:
## https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf

set -e

mkdir -p data

pushd data

# Generic test data

# wget https://www.csie.ntu.edu.tw/~cjlin/papers/guide/data/train.1
# wget https://www.csie.ntu.edu.tw/~cjlin/papers/guide/data/train.2
# wget https://www.csie.ntu.edu.tw/~cjlin/papers/guide/data/train.3

# wget https://www.csie.ntu.edu.tw/~cjlin/papers/guide/data/test.1
# wget https://www.csie.ntu.edu.tw/~cjlin/papers/guide/data/test.3

# # Leukemia

# wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/leu.bz2 && bunzip2 leu.bz2
# wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/leu.t.bz2 && bunzip2 leu.t.bz2

# Duke breast cancer

wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/duke.bz2 && bunzip2 duke.bz2

wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/duke.tr.bz2 && bunzip2 duke.tr.bz2


wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/duke.val.bz2 && bunzip2 duke.val.bz2

popd
