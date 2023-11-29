# Iris Dataset Classification with Random Forest

This project demonstrates the use of a Random Forest Classifier for the Iris dataset classification. It includes the use of `sklearn` and `hyperopt` to optimize hyperparameters.

## Overview

The project uses the Iris dataset, a widely used dataset in machine learning, consisting of 150 samples of iris flowers from three different species. The objective is to classify the species based on four features: sepal length, sepal width, petal length, and petal width.

## Implementation

### Libraries
- `sklearn` for machine learning algorithms and data processing.
- `hyperopt` for optimizing the hyperparameters of the Random Forest model.

### Random Forest Classifier

The Random Forest model is implemented with the following steps:
1. Load the Iris dataset.
2. Define a default Random Forest Classifier.
3. Measure the performance of the default model.
4. Define a hyperparameter space for optimization.
5. Use `hyperopt` to find the best hyperparameters.
6. Train and evaluate the optimized model.

### Performance Metrics

The performance of the models (default and optimized) is evaluated based on:
- Training time
- Mean accuracy score

## Results

The results include the training time and mean accuracy scores for both the default model and the optimized model.

## Usage

To run the optimization and evaluation, execute the Python script containing the provided code.

## Requirements

- Python 3
- `scikit-learn`
- `hyperopt`
- `numpy`

## License

Specify the license under which this project is available, e.g., MIT License.

## Authors

Oscar Sarabia,
Jesus Lugo.


## Acknowledgments

Mention any individuals, organizations, or resources that were helpful during the development of this project.
