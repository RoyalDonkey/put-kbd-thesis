#!/usr/bin/env python3
import rnn


if __name__ == '__main__':
    rnn.seed(0)
    model = rnn.train([
        '../../../data_balanced/main_balanced_merged.csv',
    ],
        n_layers=1
    )

    acc = rnn.test(model, [
        '../../../data_balanced/main_balanced_merged.csv',
    ])

    print("Accuracy:", acc)
