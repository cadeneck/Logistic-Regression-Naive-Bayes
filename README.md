
# Naive Bayes and Logistic Regression Implementation

This project contains the implementation of two popular machine learning algorithms: Naive Bayes and Logistic Regression. These algorithms are used for classification tasks and are implemented from scratch using NumPy.

## Naive Bayes Classifier

The `NaiveBayes` class implements the Naive Bayes classifier. This classifier assumes that the features are independent given the class label, which simplifies the computation of probabilities.

### Attributes:
- `alpha`: The smoothing parameter for Laplace smoothing.
- `n_features`: The number of features in the training data.
- `class_labels`: The unique class labels in the training data.
- `class_probabilities`: The prior probability of each class.
- `feature_probabilities`: The conditional probability of each feature given each class.

### Methods:
- `compute_class_probabilities`: Computes the prior probability of each class.
- `compute_feature_probabilities`: Computes the conditional probability of each feature given each class.
- `fit`: Fits the Naive Bayes model to the training data.
- `predict_probabilities`: Predicts the probability of each class for each test example.
- `predict`: Predicts the class for each test example.
- `evaluate`: Evaluates the model on the test data, computing the zero-one loss and squared loss.

## Logistic Regression Classifier

The `LogisticRegression` class implements the Logistic Regression classifier. This classifier uses a logistic function to model a binary dependent variable.

### Attributes:
- `w`: The weights of the model, including the bias term.
- `X`: The training data with the bias term included.
- `y`: The labels for the training data.

### Methods:
- `sigmoid`: Applies the sigmoid function to an input.
- `initialize_weights`: Initializes the weights to ones.
- `compute_gradient`: Computes the gradient of the loss function with respect to the weights.
- `fit`: Trains the logistic regression model using gradient descent.
- `predict`: Predicts the labels for the given data.
- `accuracy`: Computes the accuracy of the model.

Both classifiers are tested using doctests, ensuring their correctness. The logistic regression model also includes a method to compute the accuracy, allowing for easy evaluation of the model's performance.
