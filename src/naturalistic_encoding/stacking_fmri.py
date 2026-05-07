from cvxopt import matrix, solvers
import numpy as np
from scipy.stats import zscore

from naturalistic_encoding.ridge_tools import (
    R2,
    CV_ind,
    cross_val_ridge,
    fit_predict,
    ridge,
)

# Re-export for scripts that expect stacking_fmri API
__all__ = [
    "get_cv_indices",
    "feat_ridge_CV",
    "stacking_fmri",
    "stacking_CV_fmri",
    "CV_ind",
    "R2",
    "fit_predict",
]

solvers.options["show_progress"] = False


def get_cv_indices(n_samples, n_folds):
    """Generate cross-validation indices.

    Args:
        n_samples (int): Number of samples to generate indices for.
        n_folds (int): Number of folds to use in cross-validation.

    Returns:
        numpy.ndarray: Array of cross-validation indices with shape (n_samples,).
    """
    cv_indices = np.zeros((n_samples))
    n_items = int(np.floor(n_samples / n_folds))  # number of items in one fold
    for i in range(0, n_folds - 1):
        cv_indices[i * n_items : (i + 1) * n_items] = i
    cv_indices[(n_folds - 1) * n_items :] = n_folds - 1
    return cv_indices


def feat_ridge_CV(
    train_features,
    train_targets,
    test_features,
    method="cross_val_ridge",
    n_folds=5,
    score_function=R2,
):
    """Train a ridge regression model with cross-validation and predict on test_features."""

    if np.all(train_features == 0):
        weights = np.zeros((train_features.shape[1], train_targets.shape[1]))
        train_preds = np.zeros_like(train_targets)
    else:
        cv_indices = get_cv_indices(train_targets.shape[0], n_folds=n_folds)
        train_preds = np.zeros_like(train_targets)

        for i_cv in range(n_folds):
            train_targets_cv = np.nan_to_num(zscore(train_targets[cv_indices != i_cv]))
            train_features_cv = np.nan_to_num(
                zscore(train_features[cv_indices != i_cv])
            )
            test_features_cv = np.nan_to_num(zscore(train_features[cv_indices == i_cv]))

            if method == "simple_ridge":
                weights = ridge(train_features, train_targets, 100)
            elif method == "cross_val_ridge":
                lambdas = np.array([10**i for i in range(-6, 10)])
                if train_features.shape[1] > train_features.shape[0]:
                    weights, __ = cross_val_ridge(
                        train_features_cv,
                        train_targets_cv,
                        n_splits=5,
                        lambdas=lambdas,
                        do_plot=False,
                        method="plain",
                    )
                else:
                    weights, __ = cross_val_ridge(
                        train_features_cv,
                        train_targets_cv,
                        n_splits=5,
                        lambdas=lambdas,
                        do_plot=False,
                        method="plain",
                    )

            train_preds[cv_indices == i_cv] = test_features_cv.dot(weights)

    train_err = train_targets - train_preds

    lambdas = np.array([10**i for i in range(-6, 10)])
    weights, __ = cross_val_ridge(
        train_features,
        train_targets,
        n_splits=5,
        lambdas=lambdas,
        do_plot=False,
        method="plain",
    )

    test_preds = np.dot(test_features, weights)

    train_scores = score_function(train_preds, train_targets)
    train_variances = np.var(train_preds, axis=0)

    return train_preds, train_err, test_preds, train_scores, train_variances


def stacking_fmri(
    train_data,
    test_data,
    train_features,
    test_features,
    method="cross_val_ridge",
    score_f=R2,
):
    """Stack predictions from different feature spaces for held-out test times."""

    n_time_test = test_data.shape[0]

    assert train_data.shape[1] == test_data.shape[1]
    n_voxels = train_data.shape[1]

    assert len(train_features) == len(test_features)
    n_features = len(train_features)

    r2s = np.zeros((n_features, n_voxels))
    r2s_train = np.zeros((n_features, n_voxels))
    var_train = np.zeros((n_features, n_voxels))
    r2s_weighted = np.zeros((n_features, n_voxels))

    stacked_pred = np.zeros((n_time_test, n_voxels))
    preds_test = np.zeros((n_features, n_time_test, n_voxels))
    weighted_pred = np.zeros((n_features, n_time_test, n_voxels))

    train_data = np.nan_to_num(zscore(train_data))
    test_data = np.nan_to_num(zscore(test_data))

    train_features = [np.nan_to_num(zscore(F)) for F in train_features]
    test_features = [np.nan_to_num(zscore(F)) for F in test_features]

    err = dict()
    preds_train = dict()

    for FEATURE in range(n_features):
        (
            preds_train[FEATURE],
            error,
            preds_test[FEATURE, :, :],
            r2s_train[FEATURE, :],
            var_train[FEATURE, :],
        ) = feat_ridge_CV(
            train_features[FEATURE], train_data, test_features[FEATURE], method=method
        )
        err[FEATURE] = error

    P = np.zeros((n_voxels, n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            P[:, i, j] = np.mean(err[i] * err[j], 0)

    q = matrix(np.zeros((n_features)))
    G = matrix(-np.eye(n_features, n_features))
    h = matrix(np.zeros(n_features))
    A = matrix(np.ones((1, n_features)))
    b = matrix(np.ones(1))

    S = np.zeros((n_voxels, n_features))
    stacked_pred_train = np.zeros_like(train_data)

    for i in range(0, n_voxels):
        PP = matrix(P[i])
        S[i, :] = np.array(solvers.qp(PP, q, G, h, A, b)["x"]).reshape(n_features)

        z_test = np.array(
            [preds_test[feature_j, :, i] for feature_j in range(n_features)]
        )
        z_train = np.array(
            [preds_train[feature_j][:, i] for feature_j in range(n_features)]
        )
        stacked_pred[:, i] = np.dot(S[i, :], z_test)
        stacked_pred_train[:, i] = np.dot(S[i, :], z_train)

    stacked_train_r2s = score_f(stacked_pred_train, train_data)

    for FEATURE in range(n_features):
        weighted_pred[FEATURE, :] = preds_test[FEATURE, :] * S[:, FEATURE]

    for FEATURE in range(n_features):
        r2s[FEATURE, :] = score_f(preds_test[FEATURE], test_data)
        r2s_weighted[FEATURE, :] = score_f(weighted_pred[FEATURE], test_data)

    stacked_r2s = score_f(stacked_pred, test_data)

    return (
        r2s,
        stacked_r2s,
        r2s_weighted,
        r2s_train,
        stacked_train_r2s,
        S,
    )


def stacking_CV_fmri(data, features, method="cross_val_ridge", n_folds=5, score_f=R2):
    """Cross-validated feature stacking to predict fMRI from multiple predictors."""

    n_time, n_voxels = data.shape
    n_features = len(features)

    ind = get_cv_indices(n_time, n_folds=n_folds)

    r2s = np.zeros((n_features, n_voxels))
    r2s_train_folds = np.zeros((n_folds, n_features, n_voxels))
    var_train_folds = np.zeros((n_folds, n_features, n_voxels))
    r2s_weighted = np.zeros((n_features, n_voxels))
    stacked_train_r2s_fold = np.zeros((n_folds, n_voxels))
    stacked_pred = np.zeros((n_time, n_voxels))
    preds_test = np.zeros((n_features, n_time, n_voxels))
    weighted_pred = np.zeros((n_features, n_time, n_voxels))
    S_average = np.zeros((n_voxels, n_features))

    for ind_num in range(n_folds):
        train_ind = ind != ind_num
        test_ind = ind == ind_num
        train_data = data[train_ind]
        train_features = [F[train_ind] for F in features]
        test_data = data[test_ind]
        test_features = [F[test_ind] for F in features]

        train_data = np.nan_to_num(zscore(train_data))
        test_data = np.nan_to_num(zscore(test_data))

        train_features = [np.nan_to_num(zscore(F)) for F in train_features]
        test_features = [np.nan_to_num(zscore(F)) for F in test_features]

        err = dict()
        preds_train = dict()
        for FEATURE in range(n_features):
            (
                preds_train[FEATURE],
                error,
                preds_test[FEATURE, test_ind],
                r2s_train_folds[ind_num, FEATURE, :],
                var_train_folds[ind_num, FEATURE, :],
            ) = feat_ridge_CV(
                train_features[FEATURE],
                train_data,
                test_features[FEATURE],
                method=method,
            )
            err[FEATURE] = error

        P = np.zeros((n_voxels, n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                P[:, i, j] = np.mean(err[i] * err[j], axis=0)

        q = matrix(np.zeros((n_features)))
        G = matrix(-np.eye(n_features, n_features))
        h = matrix(np.zeros(n_features))
        A = matrix(np.ones((1, n_features)))
        b = matrix(np.ones(1))

        S = np.zeros((n_voxels, n_features))
        stacked_pred_train = np.zeros_like(train_data)

        for i in range(n_voxels):
            PP = matrix(P[i])
            S[i, :] = np.array(solvers.qp(PP, q, G, h, A, b)["x"]).reshape(
                n_features,
            )
            z = np.array(
                [preds_test[feature_j, test_ind, i] for feature_j in range(n_features)]
            )
            stacked_pred[test_ind, i] = np.dot(S[i, :], z)
            z = np.array(
                [preds_train[feature_j][:, i] for feature_j in range(n_features)]
            )
            stacked_pred_train[:, i] = np.dot(S[i, :], z)

        S_average += S
        stacked_train_r2s_fold[ind_num, :] = score_f(stacked_pred_train, train_data)

        for FEATURE in range(n_features):
            weighted_pred[FEATURE, test_ind] = (
                preds_test[FEATURE, test_ind] * S[:, FEATURE]
            )

    data_zscored = zscore(data)
    for FEATURE in range(n_features):
        r2s[FEATURE, :] = score_f(preds_test[FEATURE], data_zscored)
        r2s_weighted[FEATURE, :] = score_f(weighted_pred[FEATURE], data_zscored)

    stacked_r2s = score_f(stacked_pred, data_zscored)

    r2s_train = r2s_train_folds.mean(0)
    stacked_train = stacked_train_r2s_fold.mean(0)
    S_average = S_average / n_folds

    return (
        r2s,
        stacked_r2s,
        r2s_weighted,
        r2s_train,
        stacked_train,
        S_average,
    )
