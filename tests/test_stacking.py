import numpy as np

from naturalistic_encoding.stacking_fmri import stacking_CV_fmri


def test_stacking_cv_fmri_runs():
    rng = np.random.default_rng(0)
    n_time, n_voxels = 50, 8
    data = rng.standard_normal((n_time, n_voxels))
    f1 = rng.standard_normal((n_time, 4))
    f2 = rng.standard_normal((n_time, 3))
    features = [f1, f2]
    r2s, stacked_r2s, r2s_weighted, r2s_train, stacked_train, S = stacking_CV_fmri(
        data, features, method="cross_val_ridge", n_folds=3
    )
    assert r2s.shape == (2, n_voxels)
    assert S.shape == (n_voxels, 2)
    assert stacked_r2s.shape == (n_voxels,)
    assert np.isfinite(stacked_r2s).all()
