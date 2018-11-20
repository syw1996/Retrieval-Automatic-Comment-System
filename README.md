## Multi Attention Network：基于论文《Multiway Attention Networks for Modeling Sentence Pairs》实现的检索型自动评论系统

## Environment
- python 2.7

## Requirements
- pytorch 0.4.1
- elasticsearch
- numpy
- nltk

## Quickstart
### Step 1: Preprocess the data
python preprocess.py

The origin data files that all needed:

* `train.txt`
* `dev.txt`
* `test.txt`

After running the preprocessing, the following files are generated:

* `train.pickle`
* `dev.pickle`
* `test.pickle`
* `word-count.obj`
* `word2id.obj`

### Step 2: Training the model
python train.py --cuda

### Step 3: Test
python test.py -- cuda
