# Cross validation Balance Accuracy = 93.64 %
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

TRAIN_FILE = "dry_bean_train.csv"
TEST_FILE = "dry_bean_test.csv"
OUTPUT_FILE = "forest.csv"
N_FOLDS = 5
SEED = 2032632813
TUNE = False  # set to True to also print all the tuning tables (takes about 10 minutes)

# final forest settings, we picked them with tune()
# giving every tree more columns helped the most, 12 and 16 scored about the same
# so we keep 12 and the trees still get different columns
# more trees or deeper trees barely changed the score
N_TREES = 50
MAX_DEPTH = 15
MIN_SAMPLES = 2
N_FEATURES = 12


def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    feature_columns = [column for column in train_df.columns if column != "Class"]
    X_train = train_df[feature_columns].to_numpy(dtype=float)
    # the bean names become numbers 0 to 6, class_names[number] gives the name back
    class_names = np.unique(train_df["Class"])
    y_train = np.searchsorted(class_names, train_df["Class"])
    X_test = test_df[feature_columns].to_numpy(dtype=float)
    return X_train, y_train, class_names, test_df, X_test


def make_folds(labels, n_folds, seed):
    # shuffle every bean type and deal it out to the folds like cards, so every fold gets the same mix
    random_generator = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for class_number in np.unique(labels):
        rows_of_class = np.where(labels == class_number)[0]
        random_generator.shuffle(rows_of_class)
        for fold_number in range(n_folds):
            folds[fold_number].extend(rows_of_class[fold_number::n_folds])
    return [np.array(fold) for fold in folds]


# we print accuracy and balanced accuracy but pick models by balanced accuracy
# some bean types are much rarer than others and plain accuracy hides it when we get those wrong
def cross_validate(fit_predict, X, y, show_folds=True):
    folds = make_folds(y, N_FOLDS, SEED)
    accuracies, balanced_accuracies = [], []
    for fold_number in range(N_FOLDS):
        test_rows = folds[fold_number]
        train_rows = np.concatenate([folds[other] for other in range(N_FOLDS) if other != fold_number])
        predictions = fit_predict(X[train_rows], y[train_rows], X[test_rows])
        accuracies.append(np.mean(predictions == y[test_rows]) * 100)
        balanced_accuracies.append(balanced_accuracy_score(y[test_rows], predictions) * 100)
        if show_folds:
            print(f"fold {fold_number + 1}: acc={round(accuracies[-1], 2)} % bal_acc={round(balanced_accuracies[-1], 2)} %")
    if show_folds:
        print(f"mean:   acc={round(np.mean(accuracies), 2)} % bal_acc={round(np.mean(balanced_accuracies), 2)} %")
    return np.mean(accuracies), np.mean(balanced_accuracies)


def sklearn_tree_fit_predict(X_train, y_train, X_test, **options):
    tree = DecisionTreeClassifier(random_state=SEED, **options)
    tree.fit(X_train, y_train)
    return tree.predict(X_test)


def entropy(counts):
    # counts can also be a table with one row of class counts per cut, then we get one entropy per cut
    shares = counts / counts.sum(axis=-1, keepdims=True)
    # the slides use log base n for n classes, we use log2 since the base never changes which cut wins
    # the tiny number stops log(0), a share of 0 then adds nothing like it should
    return -(shares * np.log2(shares + 1e-12)).sum(axis=-1)


def best_split(X, y, n_classes, features):
    n_rows = len(y)
    total_counts = np.bincount(y, minlength=n_classes)
    parent_entropy = entropy(total_counts)
    # every label as a row like [0, 0, 1, 0, 0, 0, 0], adding those rows up counts the classes
    one_hot_labels = np.eye(n_classes)[y]
    best_feature, best_threshold, best_gain = None, None, 0.0
    for feature in features:
        sorted_rows = np.argsort(X[:, feature], kind="stable")
        sorted_values = X[sorted_rows, feature]
        # left_counts[i] is the class counts of the first i+1 sorted rows, the rest go right
        left_counts = np.cumsum(one_hot_labels[sorted_rows], axis=0)[:-1]
        right_counts = total_counts - left_counts
        n_left_rows = np.arange(1, n_rows)
        # information gain of every cut at once (slide 18)
        children_entropy = (n_left_rows * entropy(left_counts)
                            + (n_rows - n_left_rows) * entropy(right_counts)) / n_rows
        gains = parent_entropy - children_entropy
        # we can only cut between two different values (slide 30)
        gains[sorted_values[:-1] == sorted_values[1:]] = -1
        best_cut = np.argmax(gains)
        if gains[best_cut] > best_gain:
            best_feature = feature
            best_threshold = (sorted_values[best_cut] + sorted_values[best_cut + 1]) / 2
            best_gain = gains[best_cut]
    return best_feature, best_threshold, best_gain


def build_tree(X, y, n_classes, features, max_depth, min_samples, depth=0):
    class_counts = np.bincount(y, minlength=n_classes)
    leaf = {"leaf": True, "label": int(np.argmax(class_counts))}
    # deep trees learn the noise of the training data (slide 39), so we stop early
    if class_counts.max() == len(y) or depth >= max_depth or len(y) < min_samples:
        return leaf
    feature, threshold, _ = best_split(X, y, n_classes, features)
    if feature is None:  # no cut makes the groups any cleaner
        return leaf
    goes_left = X[:, feature] <= threshold
    return {
        "leaf": False,
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(X[goes_left], y[goes_left], n_classes, features, max_depth, min_samples, depth + 1),
        "right": build_tree(X[~goes_left], y[~goes_left], n_classes, features, max_depth, min_samples, depth + 1),
    }


def predict_tree(tree, X):
    if tree["leaf"]:
        return np.full(len(X), tree["label"])
    goes_left = X[:, tree["feature"]] <= tree["threshold"]
    predictions = np.empty(len(X), dtype=int)
    predictions[goes_left] = predict_tree(tree["left"], X[goes_left])
    predictions[~goes_left] = predict_tree(tree["right"], X[~goes_left])
    return predictions


def id3_fit_predict(X_train, y_train, X_test, max_depth=MAX_DEPTH, min_samples=MIN_SAMPLES):
    n_classes = len(np.unique(y_train))  # every training part has all bean types
    all_features = np.arange(X_train.shape[1])
    tree = build_tree(X_train, y_train, n_classes, all_features, max_depth, min_samples)
    return predict_tree(tree, X_test)


def build_forest(X, y, n_classes, n_trees, max_depth, min_samples, n_features, seed):
    # every tree gets random rows (the same row can come up twice) and its own random columns (slide 41)
    random_generator = np.random.default_rng(seed)
    forest = []
    for _ in range(n_trees):
        tree_rows = random_generator.choice(len(y), len(y), replace=True)
        tree_features = random_generator.choice(X.shape[1], n_features, replace=False)
        forest.append(build_tree(X[tree_rows], y[tree_rows], n_classes, tree_features, max_depth, min_samples))
    return forest


def predict_forest(forest, X, n_classes):
    # simple voting, the bean type with the most votes wins
    votes = np.zeros((len(X), n_classes), dtype=int)
    all_rows = np.arange(len(X))
    for tree in forest:
        votes[all_rows, predict_tree(tree, X)] += 1
    return np.argmax(votes, axis=1)


def forest_fit_predict(X_train, y_train, X_test, n_trees=N_TREES, max_depth=MAX_DEPTH,
                       min_samples=MIN_SAMPLES, n_features=N_FEATURES):
    n_classes = len(np.unique(y_train))
    forest = build_forest(X_train, y_train, n_classes, n_trees, max_depth, min_samples, n_features, SEED)
    return predict_forest(forest, X_test, n_classes)


def print_table(title, fit_predict, settings, X, y):
    print("\n" + title)
    print("setting".ljust(26), "| acc   | bal_acc")
    for name, options in settings:
        accuracy, balanced_accuracy = cross_validate(
            lambda X_train, y_train, X_test: fit_predict(X_train, y_train, X_test, **options),
            X, y, show_folds=False)
        print(name.ljust(26), "|", round(accuracy, 2), "|", round(balanced_accuracy, 2))


def tune(X, y):
    # here we tried out the tree options and the forest settings with our cross validation
    # this prints a small table for each, every row changes one thing so you can see what it does
    # it only runs with TUNE = True, we ran it once to pick the settings at the top
    print_table("(a) sklearn tree options", sklearn_tree_fit_predict, [
        ("default (gini, no limits)", {}),
        ("criterion=entropy", {"criterion": "entropy"}),
        ("max_depth=10", {"max_depth": 10}),
        ("min_samples_leaf=5", {"min_samples_leaf": 5}),
        ("max_features=sqrt", {"max_features": "sqrt"}),
        ("class_weight=balanced", {"class_weight": "balanced"}),
    ], X, y)

    # our tree and the sklearn entropy tree at the same depths should score about the same
    depths = [5, 10, 20]
    print_table("(b) our id3 tree", id3_fit_predict,
                [(f"max_depth={depth}", {"max_depth": depth}) for depth in depths], X, y)
    print_table("    sklearn entropy tree", sklearn_tree_fit_predict,
                [(f"max_depth={depth}", {"criterion": "entropy", "max_depth": depth}) for depth in depths], X, y)

    print_table("(c) our forest, every row changes one thing from the final settings", forest_fit_predict, [
        ("final settings", {}),
        ("n_trees=10", {"n_trees": 10}),
        ("n_trees=100", {"n_trees": 100}),
        ("n_features=4", {"n_features": 4}),
        ("n_features=8", {"n_features": 8}),
        ("n_features=16", {"n_features": 16}),
        ("max_depth=20", {"max_depth": 20}),
    ], X, y)


def main():
    X_train, y_train, class_names, test_df, X_test = load_data()
    if TUNE:
        tune(X_train, y_train)

    print(f"\nfinal forest: {N_TREES} trees, {N_FEATURES} features each, "
          f"max_depth={MAX_DEPTH}, min_samples={MIN_SAMPLES}")
    _, balanced_accuracy = cross_validate(forest_fit_predict, X_train, y_train)
    print(f"# Cross validation Balance Accuracy = {round(balanced_accuracy, 2)} %")

    # the cross validation above told us how good it is, so the final forest learns from all the rows
    output_df = test_df.copy()
    output_df["Target"] = class_names[forest_fit_predict(X_train, y_train, X_test)]
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"saved {OUTPUT_FILE} ({len(output_df)} rows)")


if __name__ == "__main__":
    main()
