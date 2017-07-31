import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from classify.decision_tree import ID3
from classify.decision_tree import CART


def generate_train_and_test(X):
    l = X.shape[0]
    np.random.seed(0)
    sampler = np.random.permutation(l)
    train_num = int(l / 5 * 4)
    X_train = X[sampler[:train_num]]
    X_test = X[sampler[train_num:]]
    return (X_train, X_test)

def translate_num(X):
    X_num = np.zeros_like(X)
    X_num[:, 0] = X[:, 0]
    X_num[X == 'b'] = 1
    X_num[X == 'x'] = 2
    X_num[X == 'o'] = 3
    X_num[X == 'positive'] = 1
    X_num[X == 'negative'] = 0
    return X_num

def toCSV(X, filename):
    index = np.arange(len(X))
    data = {
        'ID': index,
        'Category': X
            }
    df = pd.DataFrame(data, columns=['ID', 'Category'])
    df.to_csv(filename, index=False)

def main():
    Xtrain_filename = 'datasets/train.csv'
    ytrain_filename = 'datasets/y_train.csv'
    X = pd.read_csv(Xtrain_filename)
    labels = X.columns.tolist()
    y = pd.read_csv(ytrain_filename)
    X = np.concatenate((X.values, y['Category'][:, np.newaxis]), axis=1)
    X_train = translate_num(X)

    # Xtest_filename = 'datasets/test.csv'
    # X_test = pd.read_csv(Xtest_filename).values
    # X_test = translate_num(X_test)

    X_train, X_test = generate_train_and_test(X_train)
    print(X_train.shape)
    # print(X_test.shape)
    dt = CART(X_train, labels)
    # myTree = dt.createTree(X_train, labels)
    myTree = dt.createTree(X_train, labels)
    print (myTree)
    # myTree = {'MMS': {0: {'BLS': {0: {'TRS': {1: 1, 2: 0}}, 1: {'TMS': {0: 1, 1: {'BMS': {0: 0, 1: {'BRS': {1: 1, 2: 0}}, 2: 1}}, 2: {'TLS': {0: 1, 1: 1, 2: {'TRS': {0: 1, 2: 0}}}}}}, 2: {'TRS': {0: 0, 1: {'TLS': {0: {'TMS': {0: 0, 1: 0, 2: 1}}, 1: {'TMS': {1: 1, 2: 0}}, 2: {'TMS': {0: 0, 1: 0, 2: 1}}}}, 2: 0}}}}, 1: {'TRS': {0: {'BLS': {0: 1, 1: 1, 2: {'MLS': {0: 1, 1: {'TMS': {0: 1, 1: {'BMS': {0: 1, 1: 1, 2: 0}}, 2: 1}}, 2: {'TLS': {0: 1, 1: {'BRS': {0: 1, 1: 1, 2: 0}}, 2: 0}}}}}}, 1: {'BLS': {0: 1, 1: 1, 2: {'TMS': {0: {'BRS': {0: 1, 1: {'TLS': {0: 1, 1: 1, 2: 0}}, 2: 0}}, 1: {'BMS': {0: {'TLS': {1: 1, 2: 0}}, 1: 1, 2: 0}}, 2: {'TLS': {0: 1, 1: 1, 2: {'MLS': {1: 1, 2: 0}}}}}}}}, 2: {'MRS': {0: {'MLS': {0: {'TMS': {1: 1, 2: 0}}, 1: {'BLS': {0: 1, 2: {'BRS': {0: 1, 1: 0, 2: 0}}}}, 2: 1}}, 1: {'MLS': {0: {'BMS': {0: {'TLS': {1: 1, 2: 0}}, 1: {'TMS': {1: 1, 2: 0}}, 2: 0}}, 1: 1, 2: {'TMS': {0: 1, 1: {'BMS': {0: 0, 1: 1}}, 2: 0}}}}, 2: {'BRS': {0: {'TMS': {1: 1, 2: 0}}, 1: {'TLS': {0: 1, 1: 1, 2: {'TMS': {0: 1, 1: 0, 2: 0}}}}, 2: 0}}}}}}, 2: {'TRS': {0: {'TMS': {0: {'BRS': {0: 0, 1: {'BMS': {0: 0, 1: 1, 2: 1}}, 2: {'TLS': {1: 1, 2: 0}}}}, 1: 0, 2: {'BMS': {0: 1, 1: 1, 2: 0}}}}, 1: {'BLS': {0: {'BMS': {0: {'MRS': {0: 1, 1: {'BRS': {1: 1, 2: {'TLS': {1: 1, 2: 0}}}}, 2: {'TMS': {0: 0, 1: 1}}}}, 1: {'MRS': {0: 1, 1: {'BRS': {1: 1, 2: 0}}, 2: 0}}, 2: {'BRS': {0: 1, 1: 1, 2: {'MRS': {0: 1, 1: 0}}}}}}, 1: {'TLS': {0: {'BMS': {0: {'TMS': {0: 0, 2: 1}}, 1: {'TMS': {1: 0, 2: 1}}, 2: 0}}, 1: {'MLS': {0: {'TMS': {1: 1, 2: 0}}, 1: 1, 2: {'TMS': {0: 0, 1: 1, 2: 0}}}}, 2: {'BRS': {0: 0, 1: {'MLS': {0: 1, 1: 0, 2: 1}}, 2: 0}}}}, 2: {'MLS': {0: 1, 1: {'BMS': {0: 1, 1: {'TLS': {1: {'TMS': {1: 1, 2: 0}}, 2: {'TMS': {1: 0, 2: 1}}}}, 2: {'TLS': {0: 0, 1: 1, 2: 1}}}}, 2: {'TMS': {0: 0, 1: 1, 2: 1}}}}}}, 2: {'BLS': {0: 0, 1: {'TMS': {0: {'TLS': {0: 1, 1: 1, 2: {'MRS': {0: 1, 1: 0}}}}, 1: {'MLS': {0: 1, 1: 0, 2: {'TLS': {0: 1, 1: 0, 2: 1}}}}, 2: {'BMS': {0: 1, 1: 1, 2: 0}}}}, 2: 0}}}}}}
    y_pre_test = []
    y_test = X_test[:, -1]
    for i in X_test:
        y_i = dt.classify(myTree, labels, i)
        y_pre_test.append(y_i)
        # print y_i
    acc = float(np.sum(y_pre_test == y_test))/len(X_test)
    print(acc)
    # print(y_pre_test)
    # y_pre = ['positive'if i == 1.0 else 'negative' for i in y_pre_test]
    # pre_filename = 'results/pre.csv'
    # toCSV(y_pre, pre_filename)

def output():
    Xtrain_filename = 'datasets/train.csv'
    ytrain_filename = 'datasets/y_train.csv'
    X = pd.read_csv(Xtrain_filename)
    labels = X.columns.tolist()
    y = pd.read_csv(ytrain_filename)
    X = np.concatenate((X.values, y['Category'][:, np.newaxis]), axis=1)
    X_train = translate_num(X)

    Xtest_filename = 'datasets/test.csv'
    X_test = pd.read_csv(Xtest_filename).values
    X_test = translate_num(X_test)

    # X_train, X_test = generate_train_and_test(X_train)
    print(X_train.shape)
    # print(X_test.shape)
    dt = CART(X_train, labels)
    # myTree = dt.createTree(X_train, labels)
    myTree = dt.createTree(X_train, labels)
    print(myTree)
    # myTree = {'MMS': {0: {'BLS': {0: {'TRS': {1: 1, 2: 0}}, 1: {'TMS': {0: 1, 1: {'BMS': {0: 0, 1: {'BRS': {1: 1, 2: 0}}, 2: 1}}, 2: {'TLS': {0: 1, 1: 1, 2: {'TRS': {0: 1, 2: 0}}}}}}, 2: {'TRS': {0: 0, 1: {'TLS': {0: {'TMS': {0: 0, 1: 0, 2: 1}}, 1: {'TMS': {1: 1, 2: 0}}, 2: {'TMS': {0: 0, 1: 0, 2: 1}}}}, 2: 0}}}}, 1: {'TRS': {0: {'BLS': {0: 1, 1: 1, 2: {'MLS': {0: 1, 1: {'TMS': {0: 1, 1: {'BMS': {0: 1, 1: 1, 2: 0}}, 2: 1}}, 2: {'TLS': {0: 1, 1: {'BRS': {0: 1, 1: 1, 2: 0}}, 2: 0}}}}}}, 1: {'BLS': {0: 1, 1: 1, 2: {'TMS': {0: {'BRS': {0: 1, 1: {'TLS': {0: 1, 1: 1, 2: 0}}, 2: 0}}, 1: {'BMS': {0: {'TLS': {1: 1, 2: 0}}, 1: 1, 2: 0}}, 2: {'TLS': {0: 1, 1: 1, 2: {'MLS': {1: 1, 2: 0}}}}}}}}, 2: {'MRS': {0: {'MLS': {0: {'TMS': {1: 1, 2: 0}}, 1: {'BLS': {0: 1, 2: {'BRS': {0: 1, 1: 0, 2: 0}}}}, 2: 1}}, 1: {'MLS': {0: {'BMS': {0: {'TLS': {1: 1, 2: 0}}, 1: {'TMS': {1: 1, 2: 0}}, 2: 0}}, 1: 1, 2: {'TMS': {0: 1, 1: {'BMS': {0: 0, 1: 1}}, 2: 0}}}}, 2: {'BRS': {0: {'TMS': {1: 1, 2: 0}}, 1: {'TLS': {0: 1, 1: 1, 2: {'TMS': {0: 1, 1: 0, 2: 0}}}}, 2: 0}}}}}}, 2: {'TRS': {0: {'TMS': {0: {'BRS': {0: 0, 1: {'BMS': {0: 0, 1: 1, 2: 1}}, 2: {'TLS': {1: 1, 2: 0}}}}, 1: 0, 2: {'BMS': {0: 1, 1: 1, 2: 0}}}}, 1: {'BLS': {0: {'BMS': {0: {'MRS': {0: 1, 1: {'BRS': {1: 1, 2: {'TLS': {1: 1, 2: 0}}}}, 2: {'TMS': {0: 0, 1: 1}}}}, 1: {'MRS': {0: 1, 1: {'BRS': {1: 1, 2: 0}}, 2: 0}}, 2: {'BRS': {0: 1, 1: 1, 2: {'MRS': {0: 1, 1: 0}}}}}}, 1: {'TLS': {0: {'BMS': {0: {'TMS': {0: 0, 2: 1}}, 1: {'TMS': {1: 0, 2: 1}}, 2: 0}}, 1: {'MLS': {0: {'TMS': {1: 1, 2: 0}}, 1: 1, 2: {'TMS': {0: 0, 1: 1, 2: 0}}}}, 2: {'BRS': {0: 0, 1: {'MLS': {0: 1, 1: 0, 2: 1}}, 2: 0}}}}, 2: {'MLS': {0: 1, 1: {'BMS': {0: 1, 1: {'TLS': {1: {'TMS': {1: 1, 2: 0}}, 2: {'TMS': {1: 0, 2: 1}}}}, 2: {'TLS': {0: 0, 1: 1, 2: 1}}}}, 2: {'TMS': {0: 0, 1: 1, 2: 1}}}}}}, 2: {'BLS': {0: 0, 1: {'TMS': {0: {'TLS': {0: 1, 1: 1, 2: {'MRS': {0: 1, 1: 0}}}}, 1: {'MLS': {0: 1, 1: 0, 2: {'TLS': {0: 1, 1: 0, 2: 1}}}}, 2: {'BMS': {0: 1, 1: 1, 2: 0}}}}, 2: 0}}}}}}
    y_pre_test = []
    # y_test = X_test[:, -1]
    for i in X_test:
        y_i = dt.classify(myTree, labels, i)
        y_pre_test.append(y_i)
        # print y_i
    # acc = float(np.sum(y_pre_test == y_test))/len(X_test)
    # print(acc)
    print(y_pre_test)
    y_pre = ['positive' if i == 1.0 else 'negative' for i in y_pre_test]
    pre_filename = 'results/pre.csv'
    toCSV(y_pre, pre_filename)

def getData():
    Xtrain_filename = 'datasets/train.csv'
    ytrain_filename = 'datasets/y_train.csv'
    X = pd.read_csv(Xtrain_filename)
    labels = X.columns.tolist()
    y = pd.read_csv(ytrain_filename)
    X = np.concatenate((X.values, y['Category'][:, np.newaxis]), axis=1)
    X_train = translate_num(X)

    # Xtest_filename = 'datasets/test.csv'
    # X_test = pd.read_csv(Xtest_filename).values
    # X_test = translate_num(X_test)

    X_train, X_test = generate_train_and_test(X_train)
    print(X_train.shape)
    return X_train, X_test




if __name__ == '__main__':
    main()
    # prune()
    # output()