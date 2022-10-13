# coding: utf-8
import sys,os

import os.path
import pickle
import numpy as np


dataset_dir = os.path.dirname(os.path.abspath(__file__))
train_file = "./fer_2013_data/fer2013_train.pydict"
test_file = "./fer_2013_data/fer2013_test.pydict"

img_dim = (1, 48, 48)
img_size = 2304



def _change_one_hot_label(X,class_num = 10):
  X = np.asarray(X)
  T = np.zeros((X.size, class_num))
  for idx, row in enumerate(T):
    row[X[idx]] = 1
  return T
  
def label_smoothing(input, e=0.01):
  k = input.shape[-1]
  
  return(1-e)*input+(e/k)

def load_fer(normalize=True, flatten=False, one_hot_label=True, smooth=False, type=np.float32):
	with open(train_file, 'rb') as f:
		dataset = pickle.load(f)
	with open(test_file, 'rb') as f:
		testset = pickle.load(f)

	if normalize:
		dataset['imgs'] = dataset['imgs'].astype(type)
		dataset['imgs'] /= 255.0
		testset['imgs'] = testset['imgs'].astype(type)
		testset['imgs'] /= 255.0

	if one_hot_label:
		dataset['labels'] = _change_one_hot_label(dataset['labels'],7)
		testset['labels'] = _change_one_hot_label(testset['labels'],7)

	if not flatten:
			dataset['imgs'] = dataset['imgs'].reshape(-1, *img_dim)
			testset['imgs'] = testset['imgs'].reshape(-1, *img_dim)
			
	
	if smooth:
		dataset['labels'] = label_smoothing(dataset['labels'],0.01)
		testset['labels'] = label_smoothing(testset['labels'],0.01)
	return (dataset['imgs'], dataset['labels']),(testset['imgs'], testset['labels'])
