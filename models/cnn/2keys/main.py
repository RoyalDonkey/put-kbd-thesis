#!/usr/bin/env python3
# 2keys - a neural net to distinguish between 2 keys being pressed
import cnn

if __name__ == '__main__':
    cnn.seed(0)

    model = cnn.train([
        '../../../data/letters/k/train_2023-04-28_20-36-17.csv',
        '../../../data/letters/l/train_2023-04-28_20-37-35.csv'
    ])

    cnn.test(model, [
        '../../../data/letters/k/test_2023-04-28_20-37-18.csv',
        '../../../data/letters/l/test_2023-04-28_20-38-42.csv'
    ])
