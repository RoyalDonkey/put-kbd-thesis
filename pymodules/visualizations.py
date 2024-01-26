"""
Module for visualizing models' properties and effectiveness.
"""
import numpy as np
import matplotlib.pyplot as plt


def prediction_heatmap(predictions: np.ndarray, targets: np.ndarray,
                       alphabet: list[str], title: str) -> None:
    """Displays a heatmap confronting predictions with targets.
    Character indicated by a row was classified as the character from the column.
    """
    summary = np.zeros((len(alphabet), len(alphabet)))
    idx_dict = dict([(key, idx) for idx, key in enumerate(alphabet)])
    for target, predicted in zip(targets, predictions):
        t_idx = idx_dict[target]
        p_idx = idx_dict[predicted]
        summary[t_idx][p_idx] += 1

    plt.imshow(summary[::-1])
    plt.colorbar()
    plt.xticks(ticks=range(len(alphabet)), labels=alphabet)
    plt.yticks(ticks=range(len(alphabet)), labels=alphabet[::-1])
    plt.title(title)
    # plt.grid(color='0.5')
    plt.show()
