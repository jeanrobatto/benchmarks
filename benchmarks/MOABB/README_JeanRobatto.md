# Code changes summary

## Training

Various changes were introduced in the train.py script in order to add a series of explorative graphs to the training process.

These graphs are saved in a graphs/ subdirectory of the chosen results folder.

## Models

Two new models were added, namely:

```
EEGNetWithELA.py
ShallowConvNetWithELA.py
```

These models use the attention module defined in the ``` custom_modules ``` directory.

## Graphs

A new ``` graphing ``` python package was added, which defines all the explorative graphs added to the training script.

## Hparams search

The new models require a new hyperparameter ``` attention_kernel_size ``` which was added to the respective ``` yaml ``` files of the new models. 

This ensures that the kernel size value is included in the hyperparameter search script.

Lastly, the hyperparameter search script was tweaked in order to reduce the training time due to resource constraints.

## Notes

1. All the commits have been squashed for reading convenience.
2. All the new code is PEP8 and flake8 compliant.
3. All the previous functionality has been tested, and this code introduces no breaking changes to theproject.
4. There are no new dependencies in the project.

