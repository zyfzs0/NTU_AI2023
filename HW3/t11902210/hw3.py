from collections import Counter

import numpy as np
import pandas as pd

# set random seed
np.random.seed(0)

"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


# 1. Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    # Load iris dataset
    iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    # Load Boston housing dataset
    boston = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    return iris, boston


# 2. Preprocessing functions
def train_test_split(
    df: pd.DataFrame, target: str, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle and split dataset into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Split target and features
    X_train = train.drop(target, axis=1).values
    y_train = train[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def normalize(X: np.ndarray) -> np.ndarray:
    # Normalize features to [0, 1]
    # You can try other normalization methods, e.g., z-score, etc.
    # TODO: 1%
    # Hint: Calculate the min and max values for each feature (column).
    # Then, subtract the min value and divide by the range (max - min).
    # normalization
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_result = (X - X_min) / (X_max - X_min)
    return X_result
    # standardization(z-score)
    #X_mean = np.mean(X, axis=0)
    #X_std = np.std(X, axis=0)
    #X_result = (X - X_mean) / X_std
    #return X_result
    #raise NotImplementedError


def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels to integers.
    """
    # TODO: 1%
    # Hint: Create a dictionary that maps unique labels to integers.
    # Then, use a list comprehension to replace the original labels with their
    # corresponding integer values.
    # reference chatgpt
    labels = np.unique(y)
    label_all = {label: idx for idx, label in enumerate(labels)}
    result_labels = np.array([label_all[item] for item in y])
    return result_labels
    #raise NotImplementedError


# 3. Models
class LinearModel:
    def __init__(
        self, learning_rate=0.01, iterations=1000, model_type="linear"
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        # You can try different learning rate and iterations
        self.model_type = model_type

        assert model_type in [
            "linear",
            "logistic",
        ], "model_type must be either 'linear' or 'logistic'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.insert(X, 0, 1, axis=1)
        n_classes = len(np.unique(y))
        n_features = X.shape[1]

        if self.model_type == "logistic":
            pass
        else:
            pass
        # TODO: 2%
        # Hint: Initialize the weights based on the model type (logistic or linear).
        # Then, update the weights using gradient descent within a loop for the
        # specified number of iterations.
        if self.model_type == "logistic":
            self.weights = np.random.randn(n_features, n_classes)
            # one_hot
            one_hot = np.zeros((X.shape[0], n_classes))
            one_hot[np.arange(X.shape[0]), y.T] = 1
            y = one_hot
        else:
            self.weights = np.random.randn(n_features)
        for i in range(self.iterations):
            w_grad = self._compute_gradients(X, y)
            self.weights = self.weights - self.learning_rate * w_grad
        #raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        if self.model_type == "linear":
            # TODO: 2%
            # Hint: Perform a matrix multiplication between the input features (X)
            # and the learned weights.
            result = np.dot(X, self.weights)
            return result
            #raise NotImplementedError
        elif self.model_type == "logistic":
            # TODO: 2%
            # Hint: Perform a matrix multiplication between the input features (X)
            # and the learned weights, then apply the softmax function to the result.
            temp = np.dot(X, self.weights)
            result = self._softmax(temp)
            y_pred = np.argmax(result, axis=1)
            #y_pred = np.round(result)
            # y_pred = result
            return y_pred.astype(int)
            #raise NotImplementedError

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.model_type == "linear":
            # TODO: 3%
            # Hint: Calculate the gradients for linear regression by computing
            # the dot product of X transposed and the difference between the
            # predicted values and the true values, then normalize by the number of samples.
            m = X.shape[0]
            y_pred = X.dot(self.weights)
            return 2 * (X.T).dot(y_pred - y) / m
            #raise NotImplementedError
        elif self.model_type == "logistic":
            # TODO: 3%
            # Hint: Calculate the gradients for logistic regression by computing
            # the dot product of X transposed and the difference between the one-hot
            # encoded true values and the softmax of the predicted values,
            # then normalize by the number of samples.
            m = X.shape[0]
            h_x = X.dot(self.weights)
            y_pred = self._softmax(h_x)
            return (X.T).dot(y_pred - y) / m
            #raise NotImplementedError

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)


class DecisionTree:
    def __init__(self, max_depth: int = 5, model_type: str = "classifier"):
        self.max_depth = max_depth
        self.model_type = model_type

        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree = self._build_tree(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if depth >= self.max_depth or self._is_pure(y):
            return self._create_leaf(y)

        feature, threshold = self._find_best_split(X, y)
        # TODO: 4%
        # Hint: Create a mask based on the best feature and threshold that
        # separates the samples into two groups. Then, recursively build
        # the left and right child nodes of the current node.
        # I referred to chatgpt for this segmentation algorithm
        separate = X[:, feature] <= threshold
        left_X, right_X = X[separate], X[~separate]
        left_y, right_y = y[separate], y[~separate]

        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)
        #raise NotImplementedError


        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def _is_pure(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _create_leaf(self, y: np.ndarray):
        if self.model_type == "classifier":
            # TODO: 1%
            # Hint: For classification, return the most common class in the given samples.
            counts = np.bincount(y)
            result = np.argmax(counts)  # get the max
            return result
            #raise NotImplementedError
        else:
            # TODO: 1%
            # Hint: For regression, return the mean of the given samples.
            result = np.mean(y)
            return result
            #raise NotImplementedError

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        best_gini = float("inf")
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature])
            for i in range(1, len(X)):
                if X[sorted_indices[i - 1], feature] != X[sorted_indices[i], feature]:
                    threshold = (
                        X[sorted_indices[i - 1], feature]
                        + X[sorted_indices[i], feature]
                    ) / 2
                    mask = X[:, feature] <= threshold
                    left_y, right_y = y[mask], y[~mask]

                    if self.model_type == "classifier":
                        gini = self._gini_index(left_y, right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature = feature
                            best_threshold = threshold
                    else:
                        mse = self._mse(left_y, right_y)
                        if mse < best_mse:
                            best_mse = mse
                            best_feature = feature
                            best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        # Hint: Calculate the Gini index for the left and right samples,
        # then compute the weighted average based on the number of samples in each group.
        # gini_index reference link
        # https://blog.csdn.net/weixin_39913422/article/details/111664969
        n = len(left_y) + len(right_y)
        gini = 0.0
        for item in [left_y, right_y]:
            length = len(item)
            if length != 0:
                score = 0.0
                for e in set(item):
                    p = (item == e).sum() / length
                    score += p ** 2
                gini += (1.0 - score) * (length / n)
        return gini
        #raise NotImplementedError


    def _mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        # Hint: Calculate the mean squared error for the left and right samples,
        # then compute the weighted average based on the number of samples in each group.
        n = len(left_y) + len(right_y)
        left_mean = np.mean(left_y)
        right_mean = np.mean(right_y)
        left_sum = np.mean((left_y - left_mean) ** 2)
        right_sum = np.mean((right_y - right_mean) ** 2)
        return (left_sum * len(left_y) + right_sum * len(right_y) ) / n
        #raise NotImplementedError

    def _traverse_tree(self, x: np.ndarray, node: dict):
        if isinstance(node, dict):
            feature, threshold = node["feature"], node["threshold"]
            if x[feature] <= threshold:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        else:
            return node


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        model_type: str = "classifier",
    ):
        # TODO: 1%
        # Hint: Initialize a list of DecisionTree instances based on the
        # specified number of estimators, max depth, and model type.
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_type = model_type
        self.trees = []
        for i in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, model_type=self.model_type)
            self.trees.append(tree)
        #raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for tree in self.trees:
            # TODO: 2%
            # Hint: Generate bootstrap indices by random sampling with replacement,
            # then fit each tree with the corresponding samples from X and y.
            # bootstrap_indices = np.random.choice(
            bootstrap_indices = np.random.choice(range(len(y)), len(y), replace=True)
            now_X = X[bootstrap_indices]
            now_y = y[bootstrap_indices]
            tree.fit(now_X, now_y)
            #raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: 2%
        # Hint: Predict the output for each tree and combine the predictions
        # based on the model type (majority voting for classification or averaging
        # for regression).
        result = []
        for e in self.trees:
            result.append(e.predict(X))
        if self.model_type == "classifier":
            result0 = np.array(result)
            predictions = []
            for i in range(result0.shape[1]):
                count = np.bincount(result0[:, i])
                predictions.append(np.argmax(count))
            return np.array(predictions)
        else:
            return np.mean(result, axis=0)
        #raise NotImplementedError


# 4. Evaluation metrics
def accuracy(y_true, y_pred):
    # TODO: 1%
    # Hint: Calculate the percentage of correct predictions by comparing
    # the true and predicted labels.
    num = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            num += 1
    result = num / len(y_true)
    return result
    #raise NotImplementedError


def mean_squared_error(y_true, y_pred):
    # TODO: 1%
    # Hint: Calculate the mean squared error between the true and predicted values.
    temp = (y_true - y_pred) ** 2
    result = np.mean(temp)
    return result
    #raise NotImplementedError


# 5. Main function
def main():
    iris, boston = load_data()

    # Iris dataset - Classification
    X_train, X_test, y_train, y_test = train_test_split(iris, "class")
    X_train, X_test = normalize(X_train), normalize(X_test)
    y_train, y_test = encode_labels(y_train), encode_labels(y_test)

    logistic_regression = LinearModel(model_type="logistic")
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy(y_test, y_pred))

    #logistic_regression = LinearModel(learning_rate=0.05, iterations=1000, model_type="logistic")
    #logistic_regression.fit(X_train, y_train)
    #y_pred = logistic_regression.predict(X_test)
    #print("Logistic Regression (raise learning_rate) Accuracy:", accuracy(y_test, y_pred))

    #logistic_regression = LinearModel(learning_rate=0.01, iterations=2000, model_type="logistic")
    #logistic_regression.fit(X_train, y_train)
    #y_pred = logistic_regression.predict(X_test)
    #print("Logistic Regression (raise iterations) Accuracy:", accuracy(y_test, y_pred))

    decision_tree_classifier = DecisionTree(model_type="classifier")
    decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)
    print("Decision Tree Classifier Accuracy:", accuracy(y_test, y_pred))

    random_forest_classifier = RandomForest(model_type="classifier")
    random_forest_classifier.fit(X_train, y_train)
    y_pred = random_forest_classifier.predict(X_test)
    print("Random Forest Classifier Accuracy:", accuracy(y_test, y_pred))

    #random_forest_classifier = RandomForest(n_estimators = 500,max_depth = 5,model_type="classifier")
    #random_forest_classifier.fit(X_train, y_train)
    #y_pred = random_forest_classifier.predict(X_test)
    #print("Random Forest Classifier (raise number) Accuracy:", accuracy(y_test, y_pred))

    #random_forest_classifier = RandomForest(n_estimators = 100,max_depth = 10,model_type="classifier")
    #random_forest_classifier.fit(X_train, y_train)
    #y_pred = random_forest_classifier.predict(X_test)
    #print("Random Forest Classifier (raise depth) Accuracy:", accuracy(y_test, y_pred))

    # Boston dataset - Regression
    X_train, X_test, y_train, y_test = train_test_split(boston, "medv")
    X_train, X_test = normalize(X_train), normalize(X_test)

    linear_regression = LinearModel(model_type="linear")
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

    #linear_regression = LinearModel(learning_rate=0.002, iterations=1000,model_type="linear")
    #linear_regression.fit(X_train, y_train)
    #y_pred = linear_regression.predict(X_test)
    #print("Linear Regression (low learning_rate) MSE:", mean_squared_error(y_test, y_pred))

    #linear_regression = LinearModel(learning_rate=0.01, iterations=500,model_type="linear")
    #linear_regression.fit(X_train, y_train)
    #y_pred = linear_regression.predict(X_test)
    #print("Linear Regression (low iterations) MSE:", mean_squared_error(y_test, y_pred))

    decision_tree_regressor = DecisionTree(model_type="regressor")
    decision_tree_regressor.fit(X_train, y_train)
    y_pred = decision_tree_regressor.predict(X_test)
    print("Decision Tree Regressor MSE:", mean_squared_error(y_test, y_pred))

    random_forest_regressor = RandomForest(model_type="regressor")
    random_forest_regressor.fit(X_train, y_train)
    y_pred = random_forest_regressor.predict(X_test)
    print("Random Forest Regressor MSE:", mean_squared_error(y_test, y_pred))

    #random_forest_regressor = RandomForest(n_estimators = 50,max_depth = 5,model_type="regressor")
    #random_forest_regressor.fit(X_train, y_train)
    #y_pred = random_forest_regressor.predict(X_test)
    #print("Random Forest Regressor (low number) MSE:", mean_squared_error(y_test, y_pred))

    #random_forest_regressor = RandomForest(n_estimators = 100,max_depth = 2,model_type="regressor")
    #random_forest_regressor.fit(X_train, y_train)
    #y_pred = random_forest_regressor.predict(X_test)
    #print("Random Forest Regressor (low depth) MSE:", mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    main()
