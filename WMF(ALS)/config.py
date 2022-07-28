import argparse
import os
from os.path import join
import time

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str, required=True)
parser.add_argument("-omega", type=float, required=True)
parser.add_argument("-lambda_", type=float, required=True)
parser.add_argument("-T", type=int, required=True)
parser.add_argument("-d", type=int, required=True)
parser.add_argument("-topK", type=int, required=True)
parser.add_argument("-n", type=int, required=True)
parser.add_argument("-m", type=int, required=True)
parser.add_argument("-train_path", type=str, required=True)
parser.add_argument("-test_path", type=str, required=True)
args = parser.parse_args()

dataset = args.dataset
omega = args.omega  # weight of observed data
beta = 1  # weight of missing data
lambda_ = args.lambda_  # regularization parameter
T = args.T  # iteration number
d = args.d  # number of latent dimensions
topK = args.topK  # number of top items used to evaluate
n = args.n  # number of users
m = args.m  # number of items

DATA_PATH = "../data"
train_path = join(DATA_PATH, args.train_path)  # path of train data
test_path = join(DATA_PATH, args.test_path)  # path of test data

result_path = "./test_result"
# result_path = "./hypara_result"
if not os.path.exists(result_path):
    os.makedirs(result_path, exist_ok=True)
result_file = join(result_path, f"{dataset}_d={d}_om={omega}_reg={lambda_}_"+ time.strftime("%m-%d-%Hh%Mm%Ss")+".txt" )
