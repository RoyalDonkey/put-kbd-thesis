#!/usr/bin/env python3
import numpy as np
from sklearn.decomposition import PCA

import sys
sys.path.append("./pymodules")
from custom_read_csv import KeyPressSequence
from fft_tools import new_fft


def data_peak_type(kps: KeyPressSequence, peak_type: str,
                   apply_fft: bool = False) -> tuple[np.ndarray, np.ndarray]:
    df = kps.to_pandas()
    peak_subset = peak_type.lower()
    assert set(peak_subset) <= set("thr"), \
        f"Illegal peak type: {peak_type}; a subset of 'thr' is expected"
    col_names = [{"t": "touch", "h": "hit", "r": "release"}[c] for c in peak_subset]
    Y = np.array(df["key"])
    df = df[col_names]
    if apply_fft:
        for col_name in col_names:
            df[col_name] = df[col_name].map(new_fft)
    X = []
    for _, row in df.iterrows():
        to_append = []
        for c in col_names:
            to_append += row[c].tolist()
        X.append(to_append)

    X = np.array(X).astype(np.float32)
    X /= 2 ** 15  # LinearRegression asked for scaling or increasing max_iter
    # no suitable number of iterations was found, so this was chosen as the fix
    return X, Y


if __name__ == "__main__":
    APPLY_FFT: bool = False
    PEAK_TYPE = "thr"
    N_COMPONENTS = 3
    main_kps = KeyPressSequence("./data/balanced_merged_files.csv")
    x, y = data_peak_type(main_kps, PEAK_TYPE, APPLY_FFT)
    pca = PCA(n_components=N_COMPONENTS)
    print(f"Fitting PCA with {N_COMPONENTS} components")
    pca.fit(x)
    print("Explained variances:", pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print("Exploring components")
    top_k = 30
    for i, v in enumerate(pca.components_):
        top_k_indices = np.argsort(np.abs(v))[-top_k:][::-1]
        top_k_values = v[top_k_indices]
        print(f"PC{i+1}")
        print(f"{top_k} BIGGEST ABSOLUTE VALUES")
        print(top_k_values, np.mean(top_k_values), np.std(top_k_values))
        print("THEIR INDICES")
        print(top_k_indices, np.mean(top_k_indices), np.std(top_k_indices))
        print(f"TOUCH SUM: {np.sum(np.abs(v[:176]))} HIT SUM: {np.sum(np.abs(v[176:352]))} RELEASE SUM: {np.sum(np.abs(v[352:]))}")
    print(pca.noise_variance_)
