# The sound of typing: using Machine Learning to classify Keyboard Acoustic Emanations

BibTex citation:
```
@thesis{gkw2024soundoftyping,
    title={The sound of typing: using Machine Learning to classify Keyboard Acoustic Emanations},
    author={G{\'o}lski, Marcin and Kaszubski, Piotr and Woroch, Bart{\l}omiej},
    school = {Poznan University of Technology},
    address = {Piotrowo 3, 60-965, Pozna{\’n}, Poland},
    year={2024},
    type = {Bachelor's thesis}
}
```

This document will give a brief overview of how to recreate the results obtained
in *The sound of typing: using Machine Learning to classify Keyboard Acoustic
Emanations*.

## Extracting keystroke peaks

Run `./recdata/wav_processing.py` passing the names of the .wav files as
arguments. The script will look for a matching .keys file within the same
directory as the .wav file. To see additional options, run
`./redata/wav_processing -h`.

The files used for the experiments with peaks extracted this way are available
in a separate repository,
<https://github.com/RoyalDonkey/put-kbd-thesis-datasets>.


## Combine the dataset into a single file

Run `./recdata/merge_files.py` to aggregate the scattered .csv files created in
the previous step into a single file representing the entire dataset. The
parameters must be adjusted in the call to the `merge_files()` function. Note
that the script expects a directory structure like this:

```
dataset_root/
    digits/
        ...
    letters/
        ...
    symbols/
        ...
```

## Generate the results

Adjust the parameters within the source code of `./plot/create_results.py` to
match the desired datasets (`DATASETS_LIST`), peak types (`wave_comb`) and
preprocessing techniques (`preprocess_comb`). Select the model to use in the
section that identifies itself as appropriate (look for the comment: "you will
probably need to adjust this part to particular models"). This will generate
.json files with the accuracies the selected model achieved when being trained
and tested on every pair of datasets found on `DATASET_LIST`.
To do this for RNN, use `create_results_rnn.py` in the same way.

The plots can be recreated using other scripts found in the `./plot/` directory:
`best_acc.py`, `best_models_per_dataset.py`, `datasets_performance.py`,
`model_comparison.py`, and `model_summary,py`. Results and plots used for the
thesis can be found in `./plot/results/` and `./plot/saved_plots/`.

The data for the table showcasing top-3 rnn results was created with
`./models/rnn/multkeys/ranked_network_results.py`; the script requires an
existing model instance to use to perform the predictions.
