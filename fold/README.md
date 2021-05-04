# Fold information

In our paper, we reported the results of three-fold cross-validation, conducted by splitting the sequences into three. 

Here, sequence indices in the training and validation sets in each fold are provided. Each text file contains sequence indices, which correspond to the order of the sequence in Annotation.xml. Eg. number 0 in these files means the first sequence appearing in Annotation.xml, and number 1 is the second sequence in Annotation.xml.

Validation sets were made by randomly splitting the sequences into three. Training sets were randomly chosen 2 or 8 minutes sequences from the non-validation sets. Eg. training sets of fold 0 are randomly chosen from validation sets of fold 1 and 2.

## Test codes
[test/show_duration.py](fold/test/show_duration.py) shows the total duration of each set.
