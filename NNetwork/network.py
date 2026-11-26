# Cross validation Balance Accuracy = 94.32 %
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

TRAIN_FILE = "dry_bean_train.csv"
TEST_FILE = "dry_bean_test.csv"
OUTPUT_FILE = "network.csv"
MODEL_FILE = "network.pt"
N_FOLDS = 5
SEED = 2347463199
TUNE = False  # set to True to also print the tuning table (takes about 10 minutes)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# where tune() starts from, one hidden layer so 3 layers in total (input, hidden, output)
BASE_SETTINGS = {"hidden_sizes": [64], "learning_rate": 1e-3, "batch_size": 64,
                 "epochs": 40, "dropout": 0.0, "weight_decay": 0.0}
# final settings, we picked them with tune()
# the other settings scored about the same as the baseline so we keep the small net
# we use weight decay as our regularizer, it scored about the same as the baseline
# dropout did a bit worse, the val loss never goes up so the net is not overfitting and dropout has nothing to fix
FINAL_SETTINGS = {**BASE_SETTINGS, "weight_decay": 1e-4}


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


def standardise(X_train, X_other):
    # only the training rows give the mean and std, so the rows we test on stay unseen
    # without this the columns with big numbers would matter way more than the small ones
    column_means = X_train.mean(axis=0)
    column_stds = X_train.std(axis=0)
    return (X_train - column_means) / column_stds, (X_other - column_means) / column_stds


class BeanDataset(Dataset):
    def __init__(self, X, y):
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class BeanNet(nn.Module):
    def __init__(self, n_inputs, hidden_sizes, n_classes, dropout=0.0):
        super().__init__()
        layers = []
        previous_size = n_inputs
        for hidden_size in hidden_sizes:
            layers += [nn.Linear(previous_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_size = hidden_size
        # the last layer gives one score per bean type, the loss turns the scores into probabilities
        layers.append(nn.Linear(previous_size, n_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def cross_entropy_loss(scores, targets):
    # softmax turns the scores into probabilities, we work with the log of those
    # logsumexp is log(sum(exp(scores))) but it does not blow up when a score is big
    log_probabilities = scores - torch.logsumexp(scores, dim=1, keepdim=True)
    # take the log probability of the right bean type for every row, flip the sign and divide by the batch size
    # so the loss is the average over the batch and does not grow with bigger batches
    all_rows = torch.arange(len(targets))
    return -log_probabilities[all_rows, targets].sum() / len(targets)


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch_features, batch_labels in loader:
        batch_features, batch_labels = batch_features.to(DEVICE), batch_labels.to(DEVICE)
        optimizer.zero_grad()
        loss = cross_entropy_loss(model(batch_features), batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # we add up the batch losses and divide by the number of batches so epochs are easy to compare
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features, batch_labels = batch_features.to(DEVICE), batch_labels.to(DEVICE)
            total_loss += cross_entropy_loss(model(batch_features), batch_labels).item()
    return total_loss / len(loader)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor(X, dtype=torch.float32, device=DEVICE))
    return scores.argmax(dim=1).cpu().numpy()


def train_model(X_train, y_train, X_val, y_val, settings, run_name, show=True):
    torch.manual_seed(SEED)  # same start weights and batch order every run, so the score comes out the same
    # drop_last skips the last batch if it is not full, so we never get a batch of 1 row (batchnorm breaks on that)
    train_loader = DataLoader(BeanDataset(X_train, y_train), batch_size=settings["batch_size"],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(BeanDataset(X_val, y_val), batch_size=1024) if X_val is not None else None

    model = BeanNet(X_train.shape[1], settings["hidden_sizes"], len(np.unique(y_train)),
                    settings["dropout"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    writer = SummaryWriter("runs/" + run_name)

    for epoch in range(settings["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        writer.add_scalar("loss/train", train_loss, epoch)
        log_line = f"  epoch {epoch + 1}: train_loss={round(train_loss, 4)}"
        if val_loader is not None:
            val_loss = evaluate(model, val_loader)
            writer.add_scalar("loss/val", val_loss, epoch)
            log_line += f" val_loss={round(val_loss, 4)}"
        writer.flush()  # so tensorboard shows the new point right away
        if show and (epoch + 1) % 10 == 0:
            print(log_line)
    writer.close()
    return model


# we print accuracy and balanced accuracy in percent but pick models by balanced accuracy
# some bean types are much rarer than others and plain accuracy hides it when we get those wrong
def cross_validate(settings, X, y, run_name, show=True):
    folds = make_folds(y, N_FOLDS, SEED)
    accuracies, balanced_accuracies = [], []
    for fold_number in range(N_FOLDS):
        test_rows = folds[fold_number]
        train_rows = np.concatenate([folds[other] for other in range(N_FOLDS) if other != fold_number])
        X_train_scaled, X_test_scaled = standardise(X[train_rows], X[test_rows])
        # the test fold is only used to draw the val loss curve, the net never learns from it
        model = train_model(X_train_scaled, y[train_rows], X_test_scaled, y[test_rows],
                            settings, f"{run_name}/fold{fold_number + 1}", show)
        predictions = predict(model, X_test_scaled)
        accuracies.append(np.mean(predictions == y[test_rows]) * 100)
        balanced_accuracies.append(balanced_accuracy_score(y[test_rows], predictions) * 100)
        if show:
            print(f"fold {fold_number + 1}: acc={round(accuracies[-1], 2)} % bal_acc={round(balanced_accuracies[-1], 2)} %")
    if show:
        print(f"mean:   acc={round(np.mean(accuracies), 2)} % bal_acc={round(np.mean(balanced_accuracies), 2)} %")
    return np.mean(accuracies), np.mean(balanced_accuracies)


def tune(X, y):
    # here we tried different hyperparameters and checked if a regularizer does better than the baseline
    # every row changes one thing from BASE_SETTINGS so you can see what it does
    # the last 3 rows are the regularizer comparison, dropout, weight decay and both together
    # it only runs with TUNE = True, we ran it once to pick FINAL_SETTINGS
    changes = [
        ("baseline", {}),
        ("neurons=32", {"hidden_sizes": [32]}),
        ("neurons=128", {"hidden_sizes": [128]}),
        ("hidden=64,64", {"hidden_sizes": [64, 64]}),
        ("lr=1e-2", {"learning_rate": 1e-2}),
        ("lr=1e-4", {"learning_rate": 1e-4}),
        ("batch=32", {"batch_size": 32}),
        ("batch=256", {"batch_size": 256}),
        ("dropout=0.2", {"dropout": 0.2}),
        ("weight_decay=1e-4", {"weight_decay": 1e-4}),
        ("dropout+weight_decay", {"dropout": 0.2, "weight_decay": 1e-4}),
    ]
    print("\n" + "setting".ljust(22), "| acc   | bal_acc")
    for name, change in changes:
        accuracy, balanced_accuracy = cross_validate({**BASE_SETTINGS, **change}, X, y, "tune/" + name, show=False)
        print(name.ljust(22), "|", round(accuracy, 2), "|", round(balanced_accuracy, 2))


def main():
    # to watch the losses, run tensorboard --logdir runs
    # delete the runs folder before running again, or the old curves get drawn on top of the new ones
    print(f"device: {DEVICE}")
    X_train, y_train, class_names, test_df, X_test = load_data()
    if TUNE:
        tune(X_train, y_train)

    print(f"\nfinal network: {FINAL_SETTINGS}")
    _, balanced_accuracy = cross_validate(FINAL_SETTINGS, X_train, y_train, "final")
    print(f"# Cross validation Balance Accuracy = {round(balanced_accuracy, 2)} %")

    # the cross validation above told us how good it is, so the final net learns from all the rows
    X_train_scaled, X_test_scaled = standardise(X_train, X_test)
    model = train_model(X_train_scaled, y_train, None, None, FINAL_SETTINGS, "final/all")
    # save the weights and load them back into a fresh net to make sure that works
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"saved {MODEL_FILE}")

    loaded_model = BeanNet(X_train.shape[1], FINAL_SETTINGS["hidden_sizes"], len(class_names),
                           FINAL_SETTINGS["dropout"]).to(DEVICE)
    loaded_model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    output_df = test_df.copy()
    output_df["Target"] = class_names[predict(loaded_model, X_test_scaled)]
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"saved {OUTPUT_FILE} ({len(output_df)} rows)")


if __name__ == "__main__":
    main()
