import argparse
import os
from os.path import join
import time


parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str, required=True)
parser.add_argument("-omega", type=float, required=True)
parser.add_argument("-gamma", type=float, required=True)
parser.add_argument("-group_size", type=int, required=True)
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
alpha = 1  # user-oriented weight of user u in missing data
beta = 1  # item-oriented weight of item i in missing data
gamma = args.gamma  # learning rate
group_size = args.group_size  # number of users in a user group
lambda_ = args.lambda_  # regularization parameter
T = args.T  # iteration number
d = args.d  # number of latent dimensions
topK = args.topK  # number of top items used to evaluate
n = args.n  # number of users
m = args.m  # number of items

DATA_PATH = "../data"
train_path = join(DATA_PATH, args.train_path)  # path of train data
test_path = join(DATA_PATH, args.test_path)  # path of test data

RESULT_PATH = "./result"
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

result_file = join(RESULT_PATH, f"result_{dataset}_d={d}_lr={gamma}_gs={group_size}_"+time.strftime("%m-%d-%Hh%Mm%Ss")+".txt")
cost_file = join(RESULT_PATH, f"commCost_{dataset}_d={d}_gs={group_size}_"+time.strftime("%m-%d-%Hh%Mm%Ss")+".txt")

processes = 4

mask_bits = 64  # number of bits used to represent x
prec_bits = 20  # number of bits used to represent the fractional part of x
