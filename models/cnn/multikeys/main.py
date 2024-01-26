#!/usr/bin/env python3

import cnn

if __name__ == '__main__':
    cnn.seed(0)

    model = cnn.train([
        '../../../data/balanced_train.csv'
    ])

    cnn.test(model, [
        '../../../data/balanced_test.csv'
    ])
