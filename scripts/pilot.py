import pandas as pd
import nibabel as nb
import os
import numpy as np
import glob
# import h5py
import hcp_utils as hcp
# from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from naturalistic_encoding.stacking_fmri import stacking_CV_fmri, stacking_fmri, CV_ind, R2
#from ridge_tools import R2
import matplotlib.pyplot as plt
import seaborn as sns
import time
import naturalistic_encoding.nat_asd_utils as nat_asd_utils
import sys
import argparse
from naturalistic_encoding import hrf_tools
from naturalistic_encoding.config import EXTRA_DATA_ROOT, RESNET_FEATURES_DIR
from scipy.stats import zscore as zs
from scipy.stats import zscore
from scipy.signal import resample
 
"""
script to run pilot encoding models
Standard usage: python pilot.py -s sub-01 -p auditory -f cochresnet50pca1 -d 7 -l
"""
# Inputs:
# - subject: subject id from hbn
# - parcels: a subset of the MMP parcels eg: audio, video, audiovideo, all, custom[custom not implemented yet]
# - features: the features to predict brain data from.
# - delay: length in TRs (0.8s) to account for HRF. If using an hrf feature, set to 0
# - bootstrap: the permutation count. if a number is given here it will randomly permute features and add the number to the output filename
# - plot: to plot summary figures or not.
# - zscore: to apply zscore before regression

# Outputs:
# - saved results
#   - including list of parcels and list of feautres
#   - including how long it took to run
# - optional overview plot of run

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=str, help="input subject(s)", nargs='+', required=True)
    parser.add_argument("-p", "--parcels", type=str, help="parcels: auditory, visual, audiovisual, all_select, all, custom (WIP)", required=True)
#    parser.add_argument("-c", "--customparcels", type=str, help="parcels: audio, video, audiovideo, all_select, custom")
    parser.add_argument("-f", "--features", type=str, help="eg cochresnet50, cochresnet50pca1, cochresnet50pca200, manual", required=True)
    parser.add_argument("-d", "--delay", type=int, help="parcels: audio, video, audiovideo, all, custom", required=True)
    parser.add_argument("-b", "--bootstrap", type=int, help="bootstrap: which permutation it is", default=None)
    parser.add_argument("--ridgecv_bootstrap", type=int, help="bootstrap: which permutation it is", default=None)
    parser.add_argument('-l', '--plot', help="to make a plot or not", action='store_true')  # on/off flag
    parser.add_argument('-z', '--zscore', help="to zscore or not", action='store_true')  # on/off flag
    parser.add_argument('-y', '--zscorey', help="to zscore brain data or not", action='store_true')  # on/off flag
    parser.add_argument('-g', '--himalaya', help="to run group banded regression from himalaya instead of stacked regression", action='store_true')  # on/off flag
    parser.add_argument('-r', '--ridgecv', help="to run simple ridgecv", action='store_true')  # on/off flag
    parser.add_argument('-n', '--ridgecvnew', help="to run simple ridgecv NEW DEV", action='store_true')  # on/off flag
    parser.add_argument('-e', '--elasticnetcv', help="to run simple elasticnetcv", action='store_true')  # on/off flag
    parser.add_argument('-a', '--lassocv', help="to run simple elasticnetcv", action='store_true')  # on/off flag
    parser.add_argument('-t', '--friendstask', help="the friends task if doing friends", default=None)  # on/off flag
    parser.add_argument('-v', '--v1', help="append v1 mean timecourse to features in", action='store_true')  # on/off flag
    parser.add_argument('-o', '--arousal', help="append arousal mean timecourse to features in", action='store_true')  # on/off flag
    parser.add_argument('-m', '--r2eval', help="predict the mean timecourse", action='store_true')  # on/off flag
    parser.add_argument("--fd_thresh", type=float, help="the fd threshold for censoring", default=None)
    parser.add_argument("--fd_thresh_consec3", type=float, help="the fd threshold for censoring including 3 timepoints after peak", default=None)
    parser.add_argument("--fd_thresh_consec5", type=float, help="the fd threshold for censoring including 5 timepoints after peak", default=None)

    parser.add_argument('--skip_existing', help="to skip if the output already exists", action='store_true')  # on/off flag

    args = parser.parse_args()

    sub=args.subject[0]
    delay=args.delay
    output_directory_name='good_pilots_boot'

    
    unique_name=f'sub-{sub}_roi-{args.parcels}_feat-{args.features}_delay-{delay}' # for filename saving output
    print(f'running subject {sub}')

    if args.parcels=='all':
        Y=load_sub_brain_all(sub,delay) #load the whole brain
        parcels='all'
    else:
        parcels=select_parcels(args.parcels) # load parcel set
        atlas_indices_indices=extract_parcels(parcels) # get indices of parcels
        if args.friendstask is None:
            Y=load_sub_brain(sub,delay,atlas_indices_indices) #load brain data from selected parcels
        else:
            Y=load_sub_brain_friends(sub,args.friendstask,delay,atlas_indices_indices)
            unique_name=unique_name+'_friends'
    X,features=load_features(args.features) #load X
    
    if args.v1:
        v1_feat=np.load(f'../data/features/{sub}_DM_v1.npy')
        v1_feat=v1_feat[:X.shape[0]]
        # X: (749, 5)
        X=X[:v1_feat.shape[0],:]
        # print(f'v1: {v1_feat.shape}')
        # print(f'X: {X.shape}')
        X = np.column_stack((X, v1_feat))
        unique_name=unique_name+'_v1'
    if args.arousal:
        arousal_feat=np.load(f'../data/features/{sub}_one_percent_arousal_XCP.npy')
        arousal_feat=arousal_feat[:X.shape[0]]
        # X: (749, 5)
        X=X[:arousal_feat.shape[0],:]
        # print(f'arousal: {arousal_feat.shape}')
        # print(f'X: {X.shape}')
        X = np.column_stack((X, arousal_feat))
        # print(f'new X shape: {X.shape}')
        unique_name=unique_name+'_arousal'
    if args.zscorey:
        
        print('zscoring brain data')
        Y=zscore(Y)
        unique_name=unique_name+'_z'

    if args.ridgecv:
        print('run ridgecv') #skip the array shaping here and do differently later
    elif args.ridgecvnew:
        print('run elasticnetcv') #skip the array shaping here and do differently later
    elif args.elasticnetcv:
        print('run elasticnetcv') #skip the array shaping here and do differently later
    elif args.lassocv:
        print('run lassocv') #skip the array shaping here and do differently later
    elif args.r2eval:
        print('run r2eval') #skip the array shaping here and do differently later
    else:
        X = [array[:Y.shape[0], :] for array in X] #trim X features to the same length as the Y brain data since sometimes the run was cut short
        if args.zscore:
            X=nat_asd_utils.apply_zscore(X)
            unique_name = unique_name + f'_z'
        Y=Y[:X[0].shape[0],:]

    if args.bootstrap is None:
        print("No value was passed to args.bootstrap")
    else:
        print(f"The value passed to args.bootstrap is {args.bootstrap}, randomly permuting features X")
        for i in range(len(X)):
            np.random.shuffle(X[i])
        unique_name = unique_name + f'_bootstrap-{args.bootstrap}'

    if args.himalaya:
        #run himalaya banded regression
        from himalaya.ridge import GroupRidgeCV
        from naturalistic_encoding.stacking_fmri import get_cv_indices
        unique_name = unique_name + f'_himalaya'
        if args.skip_existing and os.path.exists(f'../{output_directory_name}/{unique_name}.npz'):
            print('output exists, return')
            return
        n_time=Y.shape[0]
        n_folds=5
        ind = get_cv_indices(n_time, n_folds=n_folds)
        data=np.copy(Y)
        #features=np.copy(X)
        feats=X
        r2_list=[]
        coef_list=[]
        for ind_num in range(n_folds):
            # split data into training and testing sets
            train_ind = ind != ind_num
            test_ind = ind == ind_num
            train_data = data[train_ind]
            train_features = [F[train_ind] for F in feats]
            test_data = data[test_ind]
            test_features = [F[test_ind] for F in feats]
            banded_ridge= GroupRidgeCV(groups="input",cv=5)
            banded_ridge.fit(train_features, train_data)
            score = banded_ridge.score(test_features, test_data)
            print("R^2 Score: ", np.mean(score))
            r2_list.append(score)
            coef_list.append(banded_ridge.coef_)
        elapsed_time=time.time() - start_time
        print(elapsed_time)
        print(f'saving results')
        S_average=np.mean(coef_list,axis=0)
        banded_r2s=np.mean(r2_list,axis=0)
        binary_parcels = [np.void(s.encode('utf-8')) for s in parcels]
        binary_features = [np.void(s.encode('utf-8')) for s in features]
        
        np.savez(f'../{output_directory_name}/{unique_name}', banded_r2s=banded_r2s, S_average=S_average, elapsed_time=elapsed_time, binary_parcels=binary_parcels, binary_features=binary_features)

    elif args.ridgecv:
        from naturalistic_encoding.stacking_fmri import get_cv_indices
        from sklearn.linear_model import RidgeCV
        from sklearn.metrics import r2_score
        unique_name = unique_name + f'_ridgecv'
        
        X = X[:Y.shape[0],:]
        Y = Y[:X.shape[0],:]
        if args.fd_thresh:
            unique_name = unique_name + f'_fd-{args.fd_thresh}'
            confounds_file=f'/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/fmriprep_23.2.0/sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_desc-confounds_timeseries.tsv'
            df = pd.read_csv(confounds_file, sep='\t')
            fd=df['framewise_displacement']
            fd_thresh_count= sum(1 for value in fd if value > args.fd_thresh)
            fd_thresh_indices=np.where(fd > args.fd_thresh)[0]
            fd_thresh_indices = fd_thresh_indices[fd_thresh_indices < X.shape[0]]
            print(f'X shape: {X.shape}')
            print(f'Y shape: {Y.shape}')
            print(f'running fd thresh={args.fd_thresh}, removing {fd_thresh_count} indices')
            X = np.delete(X, fd_thresh_indices, axis=0)
            Y = np.delete(Y, fd_thresh_indices, axis=0)
            print(f'X shape: {X.shape}')
            print(f'Y shape: {Y.shape}')
        elif args.fd_thresh_consec3:
            unique_name = unique_name + f'_fd_consec-{args.fd_thresh_consec3}'
            confounds_file=f'/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/fmriprep_23.2.0/sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_desc-confounds_timeseries.tsv'
            df = pd.read_csv(confounds_file, sep='\t')
            fd=df['framewise_displacement']
            fd_thresh_count= sum(1 for value in fd if value > args.fd_thresh_consec3)
            fd_thresh_indices=np.where(fd > args.fd_thresh_consec3)[0]
            fd_thresh_indices = np.where(fd > args.fd_thresh_consec3)[0]
            # Add the next 3 consecutive indices after each found index
            extended_indices = np.concatenate([np.arange(idx, idx+4) for idx in fd_thresh_indices])
            # Remove duplicates and keep indices within valid bounds
            fd_thresh_indices = np.unique(extended_indices[extended_indices < X.shape[0]])
            fd_thresh_count_consec=fd_thresh_indices.shape

            #fd_thresh_indices = fd_thresh_indices[fd_thresh_indices < X.shape[0]]
            print(f'X shape: {X.shape}')
            print(f'Y shape: {Y.shape}')
            print(f'running fd thresh consec 3={args.fd_thresh_consec3}, removing {fd_thresh_count_consec} indices')
            X = np.delete(X, fd_thresh_indices, axis=0)
            Y = np.delete(Y, fd_thresh_indices, axis=0)
            print(f'X shape: {X.shape}')
            print(f'Y shape: {Y.shape}')
        elif args.fd_thresh_consec5:
            unique_name = unique_name + f'_fd_consec-{args.fd_thresh_consec5}'
            confounds_file=f'/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/fmriprep_23.2.0/sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_desc-confounds_timeseries.tsv'
            df = pd.read_csv(confounds_file, sep='\t')
            fd=df['framewise_displacement']
            fd_thresh_count= sum(1 for value in fd if value > args.fd_thresh_consec5)
            fd_thresh_indices=np.where(fd > args.fd_thresh_consec5)[0]
            fd_thresh_indices = np.where(fd > args.fd_thresh_consec5)[0]
            # Add the next 3 consecutive indices after each found index
            extended_indices = np.concatenate([np.arange(idx, idx+6) for idx in fd_thresh_indices])
            # Remove duplicates and keep indices within valid bounds
            fd_thresh_indices = np.unique(extended_indices[extended_indices < X.shape[0]])
            fd_thresh_count_consec=fd_thresh_indices.shape

            #fd_thresh_indices = fd_thresh_indices[fd_thresh_indices < X.shape[0]]
            print(f'X shape: {X.shape}')
            print(f'Y shape: {Y.shape}')
            print(f'running fd thresh consec 3={args.fd_thresh_consec5}, removing {fd_thresh_count_consec} indices')
            X = np.delete(X, fd_thresh_indices, axis=0)
            Y = np.delete(Y, fd_thresh_indices, axis=0)
            print(f'X shape: {X.shape}')
            print(f'Y shape: {Y.shape}')
        #trim first 15 TRs
        X = X[15:,:]
        Y= Y[15:,:]
        if args.ridgecv_bootstrap is None:
            print("No value was passed to args.ridgecv_bootstrap")
        else:
            print(f"The value passed to args.ridgecv_bootstrap is {args.ridgecv_bootstrap}, randomly permuting features X")
            np.random.shuffle(X)
            unique_name = unique_name + f'_bootstrap-{args.ridgecv_bootstrap}'
        if args.skip_existing and os.path.exists(f'../{output_directory_name}/{unique_name}.npz'):
            print('output exists, return')
            return
        n_time=Y.shape[0]
        n_folds=10
        ind = get_cv_indices(n_time, n_folds=n_folds)
        data=np.copy(Y)
        feats=np.copy(X)
        n, v = data.shape
        p = feats.shape[1]
        ind = CV_ind(n, n_folds)
        preds_all = np.zeros_like(data)
        test_r2_list=[]
        train_r2_list=[]
        coef_list=[]
        r2_scores=[]
        for ind_num in range(n_folds):
            # split data into training and testing sets
            train_ind = ind != ind_num
            test_ind = ind == ind_num
            #train_data = data[train_ind]
            train_data = np.nan_to_num(zs(data[train_ind]))
            #train_features = feats[train_ind]#[F[train_ind] for F in features]
            train_features = np.nan_to_num(zs(feats[train_ind]))
            #test_data = data[test_ind]
            test_data = np.nan_to_num(zs(data[test_ind]))
            #test_features = feats[test_ind]#[F[test_ind] for F in features]
            test_features = np.nan_to_num(zs(feats[test_ind]))
            ridge=RidgeCV(cv=10,alphas=[0.1, 1, 10, 100, 1000])
            ridge.fit(train_features, train_data)
            test_score = ridge.score(test_features, test_data)
            train_score= ridge.score(train_features, train_data)
            y_pred = ridge.predict(test_features)
            preds_all[ind == ind_num] = y_pred
            #r2 = r2_score(test_data, y_pred, multioutput='raw_values')
            #r2_scores.append(r2)
            test_r2_list.append(test_score)
            train_r2_list.append(train_score)
            coef_list.append(ridge.coef_)
        #print(preds_all.shape, data.shape)
        R2_r2 = R2(preds_all, data)
        # skl_r2 = r2_score(preds_all, data, multioutput='raw_values')
        #skl_r2 = r2_score(preds_all, data)
        # print("old MEAN skl test R^2 Score: ", format(np.mean(test_r2_list), '.2f'))
        # print("old MEAN skl test R^2 Score: ", format(np.mean(r2_scores), '.2f'))
        # print("new skl MEAN test R^2 Score: ", format(np.mean(skl_r2), '.2f'))
        print("MEAN test R^2 Score: ", format(np.mean(R2_r2), '.2f'))
        elapsed_time=time.time() - start_time
        print(elapsed_time)
        print(f'saving results')
        print("MEAN train R^2 Score: ", format(np.mean(train_r2_list), '.2f'))
        binary_parcels = [np.void(s.encode('utf-8')) for s in parcels]
        binary_features = [np.void(s.encode('utf-8')) for s in features]
        np.savez(f'../{output_directory_name}/{unique_name}', stacked_r2s=R2_r2, test_r2_list=test_r2_list, train_r2_list=train_r2_list, coef_list=coef_list ,elapsed_time=elapsed_time, binary_parcels=binary_parcels, binary_features=binary_features)
    
    elif args.r2eval:
        from naturalistic_encoding.stacking_fmri import get_cv_indices
        # from sklearn.linear_model import RidgeCV
        from sklearn.metrics import r2_score
        unique_name = unique_name + f'_r2eval'
        if args.skip_existing and os.path.exists(f'../{output_directory_name}/{unique_name}.npz'):
            print('output exists, return')
            return
        X = X[:Y.shape[0],:]
        Y = Y[:X.shape[0],:]
        #trim first 15 TRs
        X = X[15:,:]
        Y= Y[15:,:]
        n_time=Y.shape[0]
        n_folds=10
        ind = get_cv_indices(n_time, n_folds=n_folds)
        data=np.copy(Y)
        feats=np.copy(X)
        n, v = data.shape
        p = feats.shape[1]
        ind = CV_ind(n, n_folds)
        #print(ind)
        #print(f'data shape {data.shape}')
        preds_all = np.zeros_like(data)
        #preds_all = np.zeros( (data.shape[0]) )
        print(f'preds_all {preds_all.shape}')
        test_r2_list=[]
        train_r2_list=[]
        coef_list=[]
        r2_scores=[]
        for ind_num in range(n_folds):
            #print(f'fold {ind_num}')# split data into training and testing sets
            train_ind = ind != ind_num
            test_ind = ind == ind_num
            #train_data = data[train_ind]
            train_data = np.nan_to_num(zs(data[train_ind]))
            #train_features = feats[train_ind]#[F[train_ind] for F in features]
            train_features = np.nan_to_num(zs(feats[train_ind]))
            #test_data = data[test_ind]
            test_data = np.nan_to_num(zs(data[test_ind]))
            #test_features = feats[test_ind]#[F[test_ind] for F in features]
            test_features = np.nan_to_num(zs(feats[test_ind]))
            #print(f'train_data: {train_data.shape}')
            #print(f'train_features: {train_features.shape}')
            #print(f'test_data: {test_data.shape}')
            #print(f'test_features: {test_features.shape}')
            #print(train_data.shape)
            train_data_mean=np.mean(train_data,axis=0)
            test_data_mean=np.mean(test_data,axis=0)
            preds_cv = np.zeros_like(test_data_mean)
            preds_cv[:]=train_data_mean
            print('predicting shapes',preds_cv.shape, test_data_mean.shape)
            cvr2 = R2(preds_cv, test_data_mean)
            print("MEAN test R^2 Score: ", format(np.mean(cvr2), '.2f'))

            #print(train_data_mean.shape)
            #ridge=RidgeCV(cv=10,alphas=[0.1, 1, 10, 100, 1000])
            #ridge.fit(train_features, train_data)
            #test_score = ridge.score(test_features, test_data)
            #train_score= ridge.score(train_features, train_data)
            #y_pred = ridge.predict(test_features)
            preds_all[ind == ind_num] = train_data_mean
            #r2 = r2_score(test_data, y_pred, multioutput='raw_values')
            #r2_scores.append(r2)
            #test_r2_list.append(test_score)
            #train_r2_list.append(train_score)
        preds_all=np.mean(preds_all,axis=0)
        data=np.mean(data,axis=0)
        print('predicting preds_all,data:',preds_all.shape, data.shape)

        R2_r2 = R2(preds_all, data)
        # skl_r2 = r2_score(preds_all, data, multioutput='raw_values')
        #skl_r2 = r2_score(preds_all, data)
        # print("old MEAN skl test R^2 Score: ", format(np.mean(test_r2_list), '.2f'))
        # print("old MEAN skl test R^2 Score: ", format(np.mean(r2_scores), '.2f'))
        # print("new skl MEAN test R^2 Score: ", format(np.mean(skl_r2), '.2f'))
        print("MEAN test R^2 Score: ", format(np.mean(R2_r2), '.2f'))
        elapsed_time=time.time() - start_time
        print(elapsed_time)
        #print(f'saving results')
        #print("MEAN train R^2 Score: ", format(np.mean(train_r2_list), '.2f'))
        binary_parcels = [np.void(s.encode('utf-8')) for s in parcels]
        binary_features = [np.void(s.encode('utf-8')) for s in features]
        #np.savez(f'../{output_directory_name}/{unique_name}', stacked_r2s=R2_r2,  train_r2_list=train_r2_list, elapsed_time=elapsed_time, binary_parcels=binary_parcels, binary_features=binary_features)

    
    elif args.ridgecvnew:
        from naturalistic_encoding.stacking_fmri import get_cv_indices,fit_predict
        from sklearn.linear_model import RidgeCV
        from sklearn.metrics import r2_score
        unique_name = unique_name + f'_ridgecv'
        print('X:',X.shape)
        print('Y:',Y.shape)

        # X = X[:,:Y.shape[0]]
        # Y= Y[:X.shape[1],:]
        X = X[:Y.shape[0],:]
        Y= Y[:X.shape[0],:]
        
        #trim first 20 TRs
        X = X[20:,:]
        Y= Y[20:,:]
        print('X:',X.shape)
        print('Y:',Y.shape)
        n_time=Y.shape[0]
        #n_folds=10
        #ind = get_cv_indices(n_time, n_folds=n_folds)
        data=np.copy(Y)
        feats=np.copy(X)
        
        test_r2_list=[]
        train_r2_list=[]
        coef_list=[]
        r2_scores=[]
        corrs, R2s= fit_predict(data, feats, method="plain", n_folds=10)
        print(corrs.shape, R2s.shape)
        print(np.mean(R2s))
    elif args.elasticnetcv:
        from naturalistic_encoding.stacking_fmri import get_cv_indices
        from sklearn.linear_model import MultiTaskElasticNetCV
        from sklearn.linear_model import ElasticNet
        unique_name = unique_name + f'_elasticnetcv'
        if args.skip_existing and os.path.exists(f'../{output_directory_name}/{unique_name}.npz'):
            print('output exists, return')
            return
        print('X:',X.shape)
        print('Y:',Y.shape)
        # X = X[:,:Y.shape[0]]
        # Y= Y[:X.shape[1],:]
        X = X[:Y.shape[0],:]
        Y= Y[:X.shape[0],:]
        #trim first 20 TRs
        X = X[20:,:]
        Y= Y[20:,:]
        print('X:',X.shape)
        print('Y:',Y.shape)
        n_time=Y.shape[0]
        n_folds=5
        ind = get_cv_indices(n_time, n_folds=n_folds)
        data=np.copy(Y)
        feats=np.copy(X)
        test_r2_list=[]
        train_r2_list=[]
        coef_list=[]
        for ind_num in range(n_folds):
            # split data into training and testing sets
            train_ind = ind != ind_num
            test_ind = ind == ind_num
            train_data = data[train_ind]
            train_features = feats[train_ind]#[F[train_ind] for F in features]
            test_data = data[test_ind]
            test_features = feats[test_ind]#[F[test_ind] for F in features]
            elasticnet=MultiTaskElasticNetCV()
            elasticnet.fit(train_features, train_data)
            test_score = elasticnet.score(test_features, test_data)
            train_score= elasticnet.score(train_features, train_data)
            print(f"fold {ind_num} test R^2 Score: ", format(np.mean(test_score), '.2f'))
            print(f"fold {ind_num} train R^2 Score: ", format(np.mean(train_score), '.2f'))
            test_r2_list.append(test_score)
            train_r2_list.append(train_score)
        elapsed_time=time.time() - start_time
        print(elapsed_time)
        print(f'saving results')
        print("MEAN test R^2 Score: ", format(np.mean(test_r2_list), '.2f'))
        print("MEAN train R^2 Score: ", format(np.mean(train_r2_list), '.2f'))
        binary_parcels = [np.void(s.encode('utf-8')) for s in parcels]
        binary_features = [np.void(s.encode('utf-8')) for s in features]
        np.savez(f'../{output_directory_name}/{unique_name}', test_r2_list=test_r2_list, train_r2_list=train_r2_list, elapsed_time=elapsed_time, binary_parcels=binary_parcels, binary_features=binary_features)


    elif args.lassocv:
        from naturalistic_encoding.stacking_fmri import get_cv_indices
        #from sklearn.linear_model import MultiTaskLassoCV
        from sklearn.linear_model import LassoCV
        unique_name = unique_name + f'_lassocv'
        if args.skip_existing and os.path.exists(f'../{output_directory_name}/{unique_name}.npz'):
            print('output exists, return')
            return
        print('X:',X.shape)
        print('Y:',Y.shape)
        # X = X[:,:Y.shape[0]]
        # Y= Y[:X.shape[1],:]
        X = X[:Y.shape[0],:]
        Y= Y[:X.shape[0],:]
        #trim first 20 TRs
        X = X[20:,:]
        Y= Y[20:,:]
        print('X:',X.shape)
        print('Y:',Y.shape)
        n_time=Y.shape[0]
        n_folds=5
        ind = get_cv_indices(n_time, n_folds=n_folds)
        data=np.copy(Y)
        feats=np.copy(X)
        test_r2_list=[]
        train_r2_list=[]
        coef_list=[]
        for ind_num in range(n_folds):
            # split data into training and testing sets
            train_ind = ind != ind_num
            test_ind = ind == ind_num
            train_data = data[train_ind]
            train_features = feats[train_ind]#[F[train_ind] for F in features]
            test_data = data[test_ind]
            test_features = feats[test_ind]#[F[test_ind] for F in features]
            test_r2_list_list=[]
            train_r2_list_list=[]
            for i in range(Y.shape[1]):
                #lasso = LassoCV(max_iter=10000,tol=0.001)
                lasso = LassoCV()
                lasso.fit(train_features, train_data[:, i])
                test_score = lasso.score(test_features, test_data[:, i])
                train_score= lasso.score(train_features, train_data[:, i])
                # print(f"fold {ind_num} test R^2 Score: ", format(np.mean(test_score), '.2f'))
                # print(f"fold {ind_num} train R^2 Score: ", format(np.mean(train_score), '.2f'))
                test_r2_list_list.append(test_score)
                train_r2_list_list.append(train_score)
            test_r2_list.append(np.asanyarray(test_r2_list_list))
            train_r2_list.append(np.asanyarray(train_r2_list_list))
        elapsed_time=time.time() - start_time
        print(elapsed_time)
        
        print(f'saving results')
        print("MEAN test R^2 Score: ", format(np.mean(test_r2_list), '.2f'))
        print("MEAN train R^2 Score: ", format(np.mean(train_r2_list), '.2f'))
        binary_parcels = [np.void(s.encode('utf-8')) for s in parcels]
        binary_features = [np.void(s.encode('utf-8')) for s in features]
        np.savez(f'../{output_directory_name}/{unique_name}', test_r2_list=test_r2_list, train_r2_list=train_r2_list, elapsed_time=elapsed_time, binary_parcels=binary_parcels, binary_features=binary_features)

    else:
        #run stacked regression
        print(f'starting regression')
        if args.skip_existing and os.path.exists(f'../{output_directory_name}/{unique_name}.npz'):
            print('output exists, return')
            return
        r2s, stacked_r2s, r2s_weighted, _, _, S_average = stacking_CV_fmri(Y, X, method = 'cross_val_ridge',n_folds = 5,score_f=R2)
        elapsed_time=time.time() - start_time
        print(elapsed_time)
        print(f'saving results')
        binary_parcels = [np.void(s.encode('utf-8')) for s in parcels]
        binary_features = [np.void(s.encode('utf-8')) for s in features]
        np.savez(f'../{output_directory_name}/{unique_name}', r2s=r2s, stacked_r2s=stacked_r2s, r2s_weighted=r2s_weighted, S_average=S_average, elapsed_time=elapsed_time, binary_parcels=binary_parcels, binary_features=binary_features)
        if args.plot:
            plot_violins(r2s, stacked_r2s, S_average, features, unique_name)


def load_features(feat_set):
    features_manual=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
    features_cochresnet=['input_after_preproc',
                    'conv1_relu1',
                    'maxpool1',
                    'layer1',
                    'layer2',
                    'layer3',
                    'layer4',
                    'avgpool']
    features_cochresnet_short=['input_after_preproc',
                    'conv1_relu1',
                    'maxpool1',
                    'layer1',
                    'layer2',
                    'layer3',
                    'layer4']
    features_slowfast=['blocks.1_fast',
                    'blocks.1_slow',
                    'blocks.2_fast',
                    'blocks.2_slow',
                    'blocks.3_fast',
                    'blocks.3_slow',
                    'blocks.4_fast',
                    'blocks.4_slow',
                    'blocks.5',
                    'blocks.6']
    features_resnet=['relu','maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
    if feat_set=="manual":
        X=[]
        features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
        for f in features:
            feature=np.load(f'../data/features/DM_{f}.npy')
            #print(feature.shape)
            # transformer = PCA(n_components=n_components)
            scaler = StandardScaler()
            # feature=transformer.fit_transform(feature)
            feature = scaler.fit_transform(X=feature,y=None)
            #print(feature.shape)
            feat_x = resample(feature, 750, axis=0) #resample to 1hz for now 
            X.append(feat_x)

    elif feat_set=='manualhrf_srp05':
        X=[]
        features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
        for f in features:
            feature=np.load(f'../data/features/DM_{f}.npy')
            scaler = StandardScaler()
            feature = scaler.fit_transform(X=feature,y=None)
            X.append(feature)
        X=nat_asd_utils.apply_srp(X,0.5)
        for xx in X:
            hz=xx.shape[0]/600 #703 seconds in friends
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
        X_out=[]
        for xx in X:
            xx = resample(xx, 750, axis=0) #resample to 471 TRs
            X_out.append(xx)
        X=X_out
    elif feat_set=='manualhrf_srp01':
        X=[]
        features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
        for f in features:
            feature=np.load(f'../data/features/DM_{f}.npy')
            scaler = StandardScaler()
            feature = scaler.fit_transform(X=feature,y=None)
            X.append(feature)
        X=nat_asd_utils.apply_srp(X,0.1)
        for xx in X:
            hz=xx.shape[0]/600 #703 seconds in friends
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
        X_out=[]
        for xx in X:
            xx = resample(xx, 750, axis=0) #resample to 471 TRs
            X_out.append(xx)
        X=X_out
    elif feat_set=='motion_srp05':
        from sklearn.random_projection import johnson_lindenstrauss_min_dim
        from sklearn.random_projection import SparseRandomProjection
        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/DM_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]
        eps=0.5
        scaler = StandardScaler()
        X = resample(motion_features, 750, axis=0)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        n_components=johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
        if n_components < n_samples:
            srp = SparseRandomProjection(n_components=n_components,random_state=42)
            X=srp.fit_transform( X )
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']

    elif feat_set=='mean_motion':
        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/DM_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]

        motion=np.mean(motion_features,axis=1)
        motion = resample(motion, 750, axis=0)
        scaler = StandardScaler()
        motion = scaler.fit_transform(X=motion.reshape(-1, 1),y=None)
        hz=motion.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz_NEW(motion,hz)
        features=['motion']    


    elif feat_set=="lla_lufs":        
        
        features=['lla_lufs']
        lla_lufs=np.load(f'../data/features/DM_lla_lufs.npy')
        X = zscore(lla_lufs)
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X.reshape(-1, 1),hz)

    elif feat_set=="lla_rms":        
        
        features=['lla_rms']
        lla_rms=np.load(f'../data/features/DM_lla_rms.npy')
        X = zscore(lla_rms)
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X.reshape(-1, 1),hz)

    elif feat_set=="lla_lufs_rms":        
        
        features=['lla_lufs_rms']
        lla_rms=np.load(f'../data/features/DM_lla_rms.npy')
        lla_rms = zscore(lla_rms)
        hz=lla_rms.shape[0]/600 #703 seconds in friends
        lla_rms=hrf_tools.apply_optimal_hrf_10hz(lla_rms.reshape(-1, 1),hz)

        lla_lufs=np.load(f'../data/features/DM_lla_lufs.npy')
        lla_lufs = zscore(lla_lufs)
        hz=lla_lufs.shape[0]/600 #703 seconds in friends
        lla_lufs=hrf_tools.apply_optimal_hrf_10hz(lla_lufs.reshape(-1, 1),hz)

        X = np.concatenate((lla_rms, lla_lufs), axis=1)

    elif feat_set=="lla_cochpca5":        
        
        features=['layer1']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        n_components=5
        pca = PCA(n_components=n_components)
        X=pca.fit_transform(X)
        hz=X.shape[0]/600
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)


    elif feat_set=="lla_cochpca10":        
        
        features=['layer1']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        n_components=10
        pca = PCA(n_components=n_components)
        X=pca.fit_transform(X)
        hz=X.shape[0]/600
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)


    elif feat_set=="lla_cochpca25":        
        
        features=['layer1']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        n_components=25
        pca = PCA(n_components=n_components)
        X=pca.fit_transform(X)
        hz=X.shape[0]/600
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
    
    
    elif feat_set=="lla_lufs_cochpca5":        
        
        features=['layer1']
        lla_lufs=np.load(f'../data/features/DM_lla_lufs.npy')
        lla_lufs = zscore(lla_lufs)
        hz=lla_lufs.shape[0]/600 #703 seconds in friends
        lla_lufs=hrf_tools.apply_optimal_hrf_10hz(lla_lufs.reshape(-1, 1),hz)


        lla_cochpca5=nat_asd_utils.load_audio_features('DM',features)[0]
        n_components=5
        pca = PCA(n_components=n_components)
        lla_cochpca5=pca.fit_transform(lla_cochpca5)
        hz=lla_cochpca5.shape[0]/600
        lla_cochpca5=hrf_tools.apply_optimal_hrf_10hz(lla_cochpca5,hz)
        
        X = np.concatenate((lla_lufs[:749], lla_cochpca5), axis=1)
        features=['lla_lufs_cochpca5']


    elif feat_set=="lla_lufs_cochpca5_friends_s01e02a":        
        
        features=['layer1']
        lla_lufs=np.load(f'../data/features/friends_s01e02a_lla_lufs.npy')
        #remove the -inf stuff :[
        min_value = np.min(lla_lufs[np.isfinite(lla_lufs)])
        lla_lufs[lla_lufs == -np.inf] = min_value

        
        lla_lufs = zscore(lla_lufs)
        hz=lla_lufs.shape[0]/703 #703 seconds in friends
        lla_lufs=hrf_tools.apply_optimal_hrf_10hz(lla_lufs.reshape(-1, 1),hz)


        lla_cochpca5=nat_asd_utils.load_audio_features('friends_s01e02a',features)[0]
        lla_cochpca5[np.isnan(lla_cochpca5)] = 0

        n_components=5
        pca = PCA(n_components=n_components)
        lla_cochpca5=pca.fit_transform(lla_cochpca5)
        hz=lla_cochpca5.shape[0]/703
        lla_cochpca5=hrf_tools.apply_optimal_hrf_10hz(lla_cochpca5,hz)
        
        X = np.concatenate((lla_lufs[:-1], lla_cochpca5), axis=1)
        features=['lla_lufs_cochpca5_friends_s01e02a']

    elif feat_set=="lla_lufs_cochpca5_friends_s01e02b":        
        
        features=['layer1']
        lla_lufs=np.load(f'../data/features/friends_s01e02b_lla_lufs.npy')
        #remove the -inf stuff :[
        min_value = np.min(lla_lufs[np.isfinite(lla_lufs)])
        lla_lufs[lla_lufs == -np.inf] = min_value

        
        lla_lufs = zscore(lla_lufs)
        hz=lla_lufs.shape[0]/703 #703 seconds in friends
        lla_lufs=hrf_tools.apply_optimal_hrf_10hz(lla_lufs.reshape(-1, 1),hz)


        lla_cochpca5=nat_asd_utils.load_audio_features('friends_s01e02b',features)[0]
        lla_cochpca5[np.isnan(lla_cochpca5)] = 0

        n_components=5
        pca = PCA(n_components=n_components)
        lla_cochpca5=pca.fit_transform(lla_cochpca5)
        hz=lla_cochpca5.shape[0]/703
        lla_cochpca5=hrf_tools.apply_optimal_hrf_10hz(lla_cochpca5,hz)
        
        X = np.concatenate((lla_lufs[:-1], lla_cochpca5), axis=1)
        features=['lla_lufs_cochpca5_friends_s01e02b']


    elif feat_set=="lla_lufs_rms_cochpca5":        
        features=['layer1']
        lla_rms=np.load(f'../data/features/DM_lla_rms.npy')
        lla_rms = zscore(lla_rms)
        hz=lla_rms.shape[0]/600 #703 seconds in friends
        lla_rms=hrf_tools.apply_optimal_hrf_10hz(lla_rms.reshape(-1, 1),hz)

        lla_lufs=np.load(f'../data/features/DM_lla_lufs.npy')
        lla_lufs = zscore(lla_lufs)
        hz=lla_lufs.shape[0]/600 #703 seconds in friends
        lla_lufs=hrf_tools.apply_optimal_hrf_10hz(lla_lufs.reshape(-1, 1),hz)


        lla_cochpca5=nat_asd_utils.load_audio_features('DM',features)[0]
        n_components=5
        pca = PCA(n_components=n_components)
        lla_cochpca5=pca.fit_transform(lla_cochpca5)
        hz=lla_cochpca5.shape[0]/600
        lla_cochpca5=hrf_tools.apply_optimal_hrf_10hz(lla_cochpca5,hz)
        
        X = np.concatenate((lla_rms[:749], lla_lufs[:749], lla_cochpca5), axis=1)
        features=['lla_lufs_rms_cochpca10']
    
    elif feat_set=="lla_lufs_rms_cochpca10":        
        features=['layer1']
        lla_rms=np.load(f'../data/features/DM_lla_rms.npy')
        lla_rms = zscore(lla_rms)
        hz=lla_rms.shape[0]/600 #703 seconds in friends
        lla_rms=hrf_tools.apply_optimal_hrf_10hz(lla_rms.reshape(-1, 1),hz)

        lla_lufs=np.load(f'../data/features/DM_lla_lufs.npy')
        lla_lufs = zscore(lla_lufs)
        hz=lla_lufs.shape[0]/600 #703 seconds in friends
        lla_lufs=hrf_tools.apply_optimal_hrf_10hz(lla_lufs.reshape(-1, 1),hz)


        lla_cochpca5=nat_asd_utils.load_audio_features('DM',features)[0]
        n_components=10
        pca = PCA(n_components=n_components)
        lla_cochpca5=pca.fit_transform(lla_cochpca5)
        hz=lla_cochpca5.shape[0]/600
        lla_cochpca5=hrf_tools.apply_optimal_hrf_10hz(lla_cochpca5,hz)
        
        X = np.concatenate((lla_rms[:749], lla_lufs[:749], lla_cochpca5), axis=1)
        features=['lla_lufs_rms_cochpca10']

    
    elif feat_set=="lla_lufs_rms_cochpca25":        
        
        
        features=['layer1']
        lla_rms=np.load(f'../data/features/DM_lla_rms.npy')
        lla_rms = zscore(lla_rms)
        hz=lla_rms.shape[0]/600 #703 seconds in friends
        lla_rms=hrf_tools.apply_optimal_hrf_10hz(lla_rms.reshape(-1, 1),hz)

        lla_lufs=np.load(f'../data/features/DM_lla_lufs.npy')
        lla_lufs = zscore(lla_lufs)
        hz=lla_lufs.shape[0]/600 #703 seconds in friends
        lla_lufs=hrf_tools.apply_optimal_hrf_10hz(lla_lufs.reshape(-1, 1),hz)


        lla_cochpca5=nat_asd_utils.load_audio_features('DM',features)[0]
        n_components=25
        pca = PCA(n_components=n_components)
        lla_cochpca5=pca.fit_transform(lla_cochpca5)
        hz=lla_cochpca5.shape[0]/600
        lla_cochpca5=hrf_tools.apply_optimal_hrf_10hz(lla_cochpca5,hz)
        
        X = np.concatenate((lla_rms[:749], lla_lufs[:749], lla_cochpca5), axis=1)
        features=['lla_lufs_rms_cochpca25']

    elif feat_set=="lla_hla":        

        features=['layer1']
        lla_lufs=np.load(f'../data/features/DM_lla_lufs.npy')
        lla_lufs = zscore(lla_lufs)
        hz=lla_lufs.shape[0]/600 #703 seconds in friends
        lla_lufs=hrf_tools.apply_optimal_hrf_10hz(lla_lufs.reshape(-1, 1),hz)


        lla_cochpca5=nat_asd_utils.load_audio_features('DM',features)[0]
        n_components=5
        pca = PCA(n_components=n_components)
        lla_cochpca5=pca.fit_transform(lla_cochpca5)
        hz=lla_cochpca5.shape[0]/600
        lla_cochpca5=hrf_tools.apply_optimal_hrf_10hz(lla_cochpca5,hz)
        
        X1 = np.concatenate((lla_lufs[:749], lla_cochpca5), axis=1)

        
        feature=np.load(f'../data/features/DM_as_scores.npy')
        class_names = np.load(str(EXTRA_DATA_ROOT / "yamnet_class_names.npy"))
        
        as_classes_label = np.zeros((521), dtype='S50')
        as_classes_label[0:4] = 'Human Sounds, Speech'
        as_classes_label[4:67] = 'Human Sounds, Non-Speech'
        as_classes_label[67:132] = 'Animal Sounds'
        as_classes_label[132:277] = 'Music'
        as_classes_label[277:294] = 'Natural Sounds'
        as_classes_label[294:453] = 'Sounds of Things'
        as_classes_label[453:500] = 'Source Ambiguous Sounds'
        as_classes_label[500:] = 'Channel, Environment, Background'
        
        categories = [
            'Human Sounds, Speech',
            'Human Sounds, Non-Speech',
            'Animal Sounds',
            'Music',
            'Natural Sounds',
            'Sounds of Things',
            'Source Ambiguous Sounds',
            'Channel, Environment, Background'
        ]
        
        collapsed_features = np.zeros((feature.shape[0], len(categories)))
        for i, category in enumerate(categories):
            # Get the indices of the columns corresponding to the current category
            category_indices = np.where(as_classes_label == category.encode('utf-8'))[0]
            # Sum across the relevant columns and assign it to the new collapsed feature
            collapsed_features[:, i] = np.sum(feature[:, category_indices], axis=1)
            #print(category_indices)
        
        X=np.copy(collapsed_features)
        scaler = StandardScaler()
        # feature=transformer.fit_transform(feature)
        X = scaler.fit_transform(X=X,y=None)
        hz=X.shape[0]/600 #703 seconds in friends
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
        X2 = resample(X, 750, axis=0) #resample to 471 TRs
        X2=X2[:749]
        X=[X1,X2]
        features=['lla','hla']



    elif feat_set=="hla_yamnet8":        

        feature=np.load(f'../data/features/DM_as_scores.npy')
        class_names = np.load(str(EXTRA_DATA_ROOT / "yamnet_class_names.npy"))
        
        as_classes_label = np.zeros((521), dtype='S50')
        as_classes_label[0:4] = 'Human Sounds, Speech'
        as_classes_label[4:67] = 'Human Sounds, Non-Speech'
        as_classes_label[67:132] = 'Animal Sounds'
        as_classes_label[132:277] = 'Music'
        as_classes_label[277:294] = 'Natural Sounds'
        as_classes_label[294:453] = 'Sounds of Things'
        as_classes_label[453:500] = 'Source Ambiguous Sounds'
        as_classes_label[500:] = 'Channel, Environment, Background'
        
        categories = [
            'Human Sounds, Speech',
            'Human Sounds, Non-Speech',
            'Animal Sounds',
            'Music',
            'Natural Sounds',
            'Sounds of Things',
            'Source Ambiguous Sounds',
            'Channel, Environment, Background'
        ]
        
        collapsed_features = np.zeros((feature.shape[0], len(categories)))
        for i, category in enumerate(categories):
            # Get the indices of the columns corresponding to the current category
            category_indices = np.where(as_classes_label == category.encode('utf-8'))[0]
            # Sum across the relevant columns and assign it to the new collapsed feature
            collapsed_features[:, i] = np.sum(feature[:, category_indices], axis=1)
            #print(category_indices)
        
        X=np.copy(collapsed_features)
        scaler = StandardScaler()
        # feature=transformer.fit_transform(feature)
        X = scaler.fit_transform(X=X,y=None)
        hz=X.shape[0]/600 #703 seconds in friends
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
        X = resample(X, 750, axis=0) #resample to 471 TRs
        features=categories





    elif feat_set=="hla_yamnet8_friends_s01e02a":        

        feature=np.load(f'../data/features/friends_s01e02a_as_scores.npy')
        class_names = np.load(str(EXTRA_DATA_ROOT / "yamnet_class_names.npy"))
        
        as_classes_label = np.zeros((521), dtype='S50')
        as_classes_label[0:4] = 'Human Sounds, Speech'
        as_classes_label[4:67] = 'Human Sounds, Non-Speech'
        as_classes_label[67:132] = 'Animal Sounds'
        as_classes_label[132:277] = 'Music'
        as_classes_label[277:294] = 'Natural Sounds'
        as_classes_label[294:453] = 'Sounds of Things'
        as_classes_label[453:500] = 'Source Ambiguous Sounds'
        as_classes_label[500:] = 'Channel, Environment, Background'
        
        categories = [
            'Human Sounds, Speech',
            'Human Sounds, Non-Speech',
            'Animal Sounds',
            'Music',
            'Natural Sounds',
            'Sounds of Things',
            'Source Ambiguous Sounds',
            'Channel, Environment, Background'
        ]
        
        collapsed_features = np.zeros((feature.shape[0], len(categories)))
        for i, category in enumerate(categories):
            # Get the indices of the columns corresponding to the current category
            category_indices = np.where(as_classes_label == category.encode('utf-8'))[0]
            # Sum across the relevant columns and assign it to the new collapsed feature
            collapsed_features[:, i] = np.sum(feature[:, category_indices], axis=1)
            #print(category_indices)
        
        X=np.copy(collapsed_features)
        scaler = StandardScaler()
        # feature=transformer.fit_transform(feature)
        X = scaler.fit_transform(X=X,y=None)
        hz=X.shape[0]/703 #703 seconds in friends
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
        X = resample(X, 472, axis=0) #resample to 471 TRs
        features=categories

    elif feat_set=="hla_yamnet8_friends_s01e02b":        

        feature=np.load(f'../data/features/friends_s01e02b_as_scores.npy')
        class_names = np.load(str(EXTRA_DATA_ROOT / "yamnet_class_names.npy"))
        
        as_classes_label = np.zeros((521), dtype='S50')
        as_classes_label[0:4] = 'Human Sounds, Speech'
        as_classes_label[4:67] = 'Human Sounds, Non-Speech'
        as_classes_label[67:132] = 'Animal Sounds'
        as_classes_label[132:277] = 'Music'
        as_classes_label[277:294] = 'Natural Sounds'
        as_classes_label[294:453] = 'Sounds of Things'
        as_classes_label[453:500] = 'Source Ambiguous Sounds'
        as_classes_label[500:] = 'Channel, Environment, Background'
        
        categories = [
            'Human Sounds, Speech',
            'Human Sounds, Non-Speech',
            'Animal Sounds',
            'Music',
            'Natural Sounds',
            'Sounds of Things',
            'Source Ambiguous Sounds',
            'Channel, Environment, Background'
        ]
        
        collapsed_features = np.zeros((feature.shape[0], len(categories)))
        for i, category in enumerate(categories):
            # Get the indices of the columns corresponding to the current category
            category_indices = np.where(as_classes_label == category.encode('utf-8'))[0]
            # Sum across the relevant columns and assign it to the new collapsed feature
            collapsed_features[:, i] = np.sum(feature[:, category_indices], axis=1)
            #print(category_indices)
        
        X=np.copy(collapsed_features)
        scaler = StandardScaler()
        # feature=transformer.fit_transform(feature)
        X = scaler.fit_transform(X=X,y=None)
        hz=X.shape[0]/703 #703 seconds in friends
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
        X = resample(X, 472, axis=0) #resample to 471 TRs
        features=categories



    
    elif feat_set=="llv_contrast_brightness":        
        
        features=['llv_contrast_brightness']
        llv_b=np.load(f'../data/features/DM_brightness.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        llv_b=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

        llv_c=np.load(f'../data/features/DM_contrast.npy')
        llv_c = zscore(llv_c)
        hz=llv_c.shape[0]/600 #703 seconds in friends
        llv_c=hrf_tools.apply_optimal_hrf_10hz(llv_c.reshape(-1, 1),hz)
        
        X = np.concatenate((llv_b, llv_c), axis=1)

    elif feat_set=="llv_contrast_brightness_color":        
        
        features=['llv_contrast_brightness_color']
        llv_b=np.load(f'../data/features/DM_brightness.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        llv_b=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

        llv_c=np.load(f'../data/features/DM_contrast.npy')
        llv_c = zscore(llv_c)
        hz=llv_c.shape[0]/600 #703 seconds in friends
        llv_c=hrf_tools.apply_optimal_hrf_10hz(llv_c.reshape(-1, 1),hz)

        llv_cc=np.load(f'../data/features/DM_color.npy')
        llv_cc = zscore(llv_c)
        hz=llv_cc.shape[0]/600 #703 seconds in friends
        llv_cc=hrf_tools.apply_optimal_hrf_10hz(llv_cc,hz)
        
        X = np.concatenate((llv_b, llv_c,llv_cc), axis=1)

        
    elif feat_set=="llv_brightness_pliers":
        
        features=['llv_brightness2']
        llv_b=np.load(f'../data/features/DM_brightness_pliers.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
    elif feat_set=="llv_brightness2":
        
        features=['llv_brightness2']
        llv_b=np.load(f'../data/features/DM_brightness2.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

    elif feat_set=="llv_brightness3":
        features=['llv_brightness3']
        llv_b=np.load(f'../data/features/DM_brightness3.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

    elif feat_set=="llv_brightness4":
        features=['llv_brightness4']
        llv_b=np.load(f'../data/features/DM_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

    elif feat_set=="optic_flow":
        features=['optic_flow']
        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['motion'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X = resample(X, 750, axis=0)

    elif feat_set=="body":
        features=['body']
        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['body'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X = resample(X, 750, axis=0)
        
    elif feat_set=="faces":
        features=['faces']
        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['faces'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X = resample(X, 750, axis=0)


    elif feat_set=="body_faces":
        features=['body_faces']
        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['body'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X1 = resample(X, 750, axis=0)
        
        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['faces'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X2 = resample(X, 750, axis=0)
        X=np.concatenate((X1, X2), axis=1)


    
    elif feat_set=="llv_hlv":        
        llv_b=np.load(f'../data/features/DM_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X1=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['motion'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X2 = resample(X, 750, axis=0)

        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/DM_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]

        motion=np.mean(motion_features,axis=1)
        motion = resample(motion, 750, axis=0)
        scaler = StandardScaler()
        motion = scaler.fit_transform(X=motion.reshape(-1, 1),y=None)
        hz=motion.shape[0]/600 #703 seconds in friends
        X3=hrf_tools.apply_optimal_hrf_10hz_NEW(motion,hz)
        
        llv_b=np.load(f'../data/features/DM_vibrance.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X4=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
        llv_b=np.load(f'../data/features/DM_sharpness.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X5=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

        X_1 = np.concatenate((X1, X2, X3, X4, X5), axis=1)


        features=['yolo11x8_resamp']
        x=np.load(f'../data/features/yolo11x8_DM_resamp.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X1=hrf_tools.apply_optimal_hrf_10hz(x,hz)

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['body'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X2 = resample(X, 750, axis=0)

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['faces'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X3 = resample(X, 750, axis=0)

        X_2 = np.concatenate((X1, X2, X3), axis=1)
        X=[X_1,X_2]
        features=['llv','hlv']

    # WEN ET AL
    # a feature for stacked regression where each space has a layer
    elif feat_set == "wen_resnet50_stacked":
        X=[]
        features=['relu','maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
        for f in features:
            X_=np.load(str(RESNET_FEATURES_DIR / f'{f}_99.npz'))
            X.append(X_['X'])

    elif feat_set == "wen_resnet50_stacked_1":
        X=[]
        features=['relu','maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
        for f in features:
            X_=np.load(str(RESNET_FEATURES_DIR / f'{f}_1.npz'))
            X.append(X_['X'])

    elif feat_set == "wen_resnet50_stacked_5":
        X=[]
        features=['relu','maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
        for f in features:
            X_=np.load(str(RESNET_FEATURES_DIR / f'{f}_5.npz'))
            X.append(X_['X'])

    elif feat_set == "wen_resnet50_stacked_10":
        X=[]
        features=['relu','maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
        for f in features:
            X_=np.load(str(RESNET_FEATURES_DIR / f'{f}_10.npz'))
            X.append(X_['X'])

    elif feat_set == "wen_resnet50_stacked_20":
        X=[]
        features=['relu','maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
        for f in features:
            X_=np.load(str(RESNET_FEATURES_DIR / f'{f}_20.npz'))
            X.append(X_['X'])

    # a combined feature for ridgecv 
    elif feat_set == "wen_resnet50":
        features=['wen_resnet50']
        X=np.load(str(RESNET_FEATURES_DIR / 'all_99_99.npz'))
        X=X['X']
    
    elif feat_set == "llv_hlv2_lla_hla":
        llv_b=np.load(f'../data/features/DM_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X1=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        X1=X1[:749]

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['motion'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X2 = resample(X, 750, axis=0)
        X2=X2[:749]

        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/DM_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]

        motion=np.mean(motion_features,axis=1)
        motion = resample(motion, 750, axis=0)
        scaler = StandardScaler()
        motion = scaler.fit_transform(X=motion.reshape(-1, 1),y=None)
        hz=motion.shape[0]/600 #703 seconds in friends
        X3=hrf_tools.apply_optimal_hrf_10hz_NEW(motion,hz)
        X3=X3[:749]

        llv_b=np.load(f'../data/features/DM_vibrance.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X4=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        X4=X4[:749]
        
        llv_b=np.load(f'../data/features/DM_sharpness.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X5=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        X5=X5[:749]

        X_1 = np.concatenate((X1, X2, X3, X4, X5), axis=1)

        
        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['body'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X2 = resample(X, 750, axis=0)
        X2=X2[:749]

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['faces'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X3 = resample(X, 750, axis=0)
        X3=X3[:749]

        X_2 = np.concatenate((X2, X3), axis=1)



        features=['layer1']
        lla_lufs=np.load(f'../data/features/DM_lla_lufs.npy')
        lla_lufs = zscore(lla_lufs)
        hz=lla_lufs.shape[0]/600 #703 seconds in friends
        lla_lufs=hrf_tools.apply_optimal_hrf_10hz(lla_lufs.reshape(-1, 1),hz)


        lla_cochpca5=nat_asd_utils.load_audio_features('DM',features)[0]
        n_components=5
        pca = PCA(n_components=n_components)
        lla_cochpca5=pca.fit_transform(lla_cochpca5)
        hz=lla_cochpca5.shape[0]/600
        lla_cochpca5=hrf_tools.apply_optimal_hrf_10hz(lla_cochpca5,hz)
        
        X_3 = np.concatenate((lla_lufs[:749], lla_cochpca5), axis=1)

        
        feature=np.load(f'../data/features/DM_as_scores.npy')
        class_names = np.load(str(EXTRA_DATA_ROOT / "yamnet_class_names.npy"))
        
        as_classes_label = np.zeros((521), dtype='S50')
        as_classes_label[0:4] = 'Human Sounds, Speech'
        as_classes_label[4:67] = 'Human Sounds, Non-Speech'
        as_classes_label[67:132] = 'Animal Sounds'
        as_classes_label[132:277] = 'Music'
        as_classes_label[277:294] = 'Natural Sounds'
        as_classes_label[294:453] = 'Sounds of Things'
        as_classes_label[453:500] = 'Source Ambiguous Sounds'
        as_classes_label[500:] = 'Channel, Environment, Background'
        
        categories = [
            'Human Sounds, Speech',
            'Human Sounds, Non-Speech',
            'Animal Sounds',
            'Music',
            'Natural Sounds',
            'Sounds of Things',
            'Source Ambiguous Sounds',
            'Channel, Environment, Background'
        ]
        
        collapsed_features = np.zeros((feature.shape[0], len(categories)))
        for i, category in enumerate(categories):
            # Get the indices of the columns corresponding to the current category
            category_indices = np.where(as_classes_label == category.encode('utf-8'))[0]
            # Sum across the relevant columns and assign it to the new collapsed feature
            collapsed_features[:, i] = np.sum(feature[:, category_indices], axis=1)
            #print(category_indices)
        
        X=np.copy(collapsed_features)
        scaler = StandardScaler()
        # feature=transformer.fit_transform(feature)
        X = scaler.fit_transform(X=X,y=None)
        hz=X.shape[0]/600 #703 seconds in friends
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
        X4 = resample(X, 750, axis=0) #resample to 471 TRs
        X_4=X4[:749]
        X=[X_1,X_2,X_3,X_4]
        features=['llv','hlv','lla','hla']


    elif feat_set == "visual_audio":
        llv_b=np.load(f'../data/features/DM_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X1=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        X1=X1[:749]

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['motion'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X2 = resample(X, 750, axis=0)
        X2=X2[:749]

        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/DM_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]

        motion=np.mean(motion_features,axis=1)
        motion = resample(motion, 750, axis=0)
        scaler = StandardScaler()
        motion = scaler.fit_transform(X=motion.reshape(-1, 1),y=None)
        hz=motion.shape[0]/600 #703 seconds in friends
        X3=hrf_tools.apply_optimal_hrf_10hz_NEW(motion,hz)
        X3=X3[:749]

        llv_b=np.load(f'../data/features/DM_vibrance.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X4=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        X4=X4[:749]
        
        llv_b=np.load(f'../data/features/DM_sharpness.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X5=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        X5=X5[:749]

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['body'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X6 = resample(X, 750, axis=0)
        X6=X6[:749]

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['faces'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X7 = resample(X, 750, axis=0)
        X7=X7[:749]

        X_1 = np.concatenate((X1, X2, X3, X4, X5, X6, X7), axis=1)



        features=['layer1']
        lla_lufs=np.load(f'../data/features/DM_lla_lufs.npy')
        lla_lufs = zscore(lla_lufs)
        hz=lla_lufs.shape[0]/600 #703 seconds in friends
        lla_lufs=hrf_tools.apply_optimal_hrf_10hz(lla_lufs.reshape(-1, 1),hz)


        lla_cochpca5=nat_asd_utils.load_audio_features('DM',features)[0]
        n_components=5
        pca = PCA(n_components=n_components)
        lla_cochpca5=pca.fit_transform(lla_cochpca5)
        hz=lla_cochpca5.shape[0]/600
        lla_cochpca5=hrf_tools.apply_optimal_hrf_10hz(lla_cochpca5,hz)
        

        
        feature=np.load(f'../data/features/DM_as_scores.npy')
        class_names = np.load(str(EXTRA_DATA_ROOT / "yamnet_class_names.npy"))
        
        as_classes_label = np.zeros((521), dtype='S50')
        as_classes_label[0:4] = 'Human Sounds, Speech'
        as_classes_label[4:67] = 'Human Sounds, Non-Speech'
        as_classes_label[67:132] = 'Animal Sounds'
        as_classes_label[132:277] = 'Music'
        as_classes_label[277:294] = 'Natural Sounds'
        as_classes_label[294:453] = 'Sounds of Things'
        as_classes_label[453:500] = 'Source Ambiguous Sounds'
        as_classes_label[500:] = 'Channel, Environment, Background'
        
        categories = [
            'Human Sounds, Speech',
            'Human Sounds, Non-Speech',
            'Animal Sounds',
            'Music',
            'Natural Sounds',
            'Sounds of Things',
            'Source Ambiguous Sounds',
            'Channel, Environment, Background'
        ]
        
        collapsed_features = np.zeros((feature.shape[0], len(categories)))
        for i, category in enumerate(categories):
            # Get the indices of the columns corresponding to the current category
            category_indices = np.where(as_classes_label == category.encode('utf-8'))[0]
            # Sum across the relevant columns and assign it to the new collapsed feature
            collapsed_features[:, i] = np.sum(feature[:, category_indices], axis=1)
            #print(category_indices)
        
        X=np.copy(collapsed_features)
        scaler = StandardScaler()
        # feature=transformer.fit_transform(feature)
        X = scaler.fit_transform(X=X,y=None)
        hz=X.shape[0]/600 #703 seconds in friends
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
        X4 = resample(X, 750, axis=0) #resample to 471 TRs

        X_2 = np.concatenate((lla_lufs[:749], lla_cochpca5, X4[:749]), axis=1)

        X=[X_1,X_2]
        features=['visual','audio']





    
        


    elif feat_set=="llv_hlv2_lla_hla_concat":        
        llv_b=np.load(f'../data/features/DM_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X1=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        X1=X1[:749]

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['motion'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X2 = resample(X, 750, axis=0)
        X2=X2[:749]

        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/DM_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]

        motion=np.mean(motion_features,axis=1)
        motion = resample(motion, 750, axis=0)
        scaler = StandardScaler()
        motion = scaler.fit_transform(X=motion.reshape(-1, 1),y=None)
        hz=motion.shape[0]/600 #703 seconds in friends
        X3=hrf_tools.apply_optimal_hrf_10hz_NEW(motion,hz)
        X3=X3[:749]

        llv_b=np.load(f'../data/features/DM_vibrance.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X4=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        X4=X4[:749]
        
        llv_b=np.load(f'../data/features/DM_sharpness.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X5=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        X5=X5[:749]
        
        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['body'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X6 = resample(X, 750, axis=0)
        X6=X6[:749]

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['faces'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X7 = resample(X, 750, axis=0)
        X7=X7[:749]




        features=['layer1']
        lla_lufs=np.load(f'../data/features/DM_lla_lufs.npy')
        lla_lufs = zscore(lla_lufs)
        hz=lla_lufs.shape[0]/600 #703 seconds in friends
        lla_lufs=hrf_tools.apply_optimal_hrf_10hz(lla_lufs.reshape(-1, 1),hz)


        lla_cochpca5=nat_asd_utils.load_audio_features('DM',features)[0]
        n_components=5
        pca = PCA(n_components=n_components)
        lla_cochpca5=pca.fit_transform(lla_cochpca5)
        hz=lla_cochpca5.shape[0]/600
        lla_cochpca5=hrf_tools.apply_optimal_hrf_10hz(lla_cochpca5,hz)
        
        X8 = np.concatenate((lla_lufs[:749], lla_cochpca5), axis=1)

        
        feature=np.load(f'../data/features/DM_as_scores.npy')
        class_names = np.load(str(EXTRA_DATA_ROOT / "yamnet_class_names.npy"))
        
        as_classes_label = np.zeros((521), dtype='S50')
        as_classes_label[0:4] = 'Human Sounds, Speech'
        as_classes_label[4:67] = 'Human Sounds, Non-Speech'
        as_classes_label[67:132] = 'Animal Sounds'
        as_classes_label[132:277] = 'Music'
        as_classes_label[277:294] = 'Natural Sounds'
        as_classes_label[294:453] = 'Sounds of Things'
        as_classes_label[453:500] = 'Source Ambiguous Sounds'
        as_classes_label[500:] = 'Channel, Environment, Background'
        
        categories = [
            'Human Sounds, Speech',
            'Human Sounds, Non-Speech',
            'Animal Sounds',
            'Music',
            'Natural Sounds',
            'Sounds of Things',
            'Source Ambiguous Sounds',
            'Channel, Environment, Background'
        ]
        
        collapsed_features = np.zeros((feature.shape[0], len(categories)))
        for i, category in enumerate(categories):
            # Get the indices of the columns corresponding to the current category
            category_indices = np.where(as_classes_label == category.encode('utf-8'))[0]
            # Sum across the relevant columns and assign it to the new collapsed feature
            collapsed_features[:, i] = np.sum(feature[:, category_indices], axis=1)
            #print(category_indices)
        
        X=np.copy(collapsed_features)
        scaler = StandardScaler()
        # feature=transformer.fit_transform(feature)
        X = scaler.fit_transform(X=X,y=None)
        hz=X.shape[0]/600 #703 seconds in friends
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
        X9 = resample(X, 750, axis=0) #resample to 471 TRs
        X9=X9[:749]
        # X=[X1,X2,X3,X4]
        X = np.concatenate((X1,X2,X3,X4,X5,X6,X7,X8,X9), axis=1)
        features=['llv','hlv','lla','hla']



    

    
    elif feat_set=="llv_hlv2":        
        llv_b=np.load(f'../data/features/DM_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X1=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['motion'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X2 = resample(X, 750, axis=0)

        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/DM_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]

        motion=np.mean(motion_features,axis=1)
        motion = resample(motion, 750, axis=0)
        scaler = StandardScaler()
        motion = scaler.fit_transform(X=motion.reshape(-1, 1),y=None)
        hz=motion.shape[0]/600 #703 seconds in friends
        X3=hrf_tools.apply_optimal_hrf_10hz_NEW(motion,hz)
        
        llv_b=np.load(f'../data/features/DM_vibrance.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X4=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
        llv_b=np.load(f'../data/features/DM_sharpness.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X5=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

        X_1 = np.concatenate((X1, X2, X3, X4, X5), axis=1)

        
        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['body'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X2 = resample(X, 750, axis=0)

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['faces'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X3 = resample(X, 750, axis=0)

        X_2 = np.concatenate((X2, X3), axis=1)
        X=[X_1,X_2]
        features=['llv','hlv']

    elif feat_set=="llv3_hlv2":        
        llv_b=np.load(f'../data/features/DM_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X1=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/DM_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]

        motion=np.mean(motion_features,axis=1)
        motion = resample(motion, 750, axis=0)
        scaler = StandardScaler()
        motion = scaler.fit_transform(X=motion.reshape(-1, 1),y=None)
        hz=motion.shape[0]/600 #703 seconds in friends
        X2=hrf_tools.apply_optimal_hrf_10hz_NEW(motion,hz)

        X_1 = np.concatenate((X1, X2), axis=1)

        
        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['body'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X2 = resample(X, 750, axis=0)

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['faces'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X3 = resample(X, 750, axis=0)

        X_2 = np.concatenate((X2, X3), axis=1)
        X=[X_1,X_2]
        features=['llv','hlv']


    elif feat_set=="llv2_hlv2":        
        llv_b=np.load(f'../data/features/DM_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X1=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
        llv_b=np.load(f'../data/features/DM_vibrance.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X4=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
        llv_b=np.load(f'../data/features/DM_sharpness.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X5=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

        X_1 = np.concatenate((X1, X4, X5), axis=1)

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['body'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X2 = resample(X, 750, axis=0)

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['faces'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X3 = resample(X, 750, axis=0)

        X_2 = np.concatenate((X2, X3), axis=1)
        X=[X_1,X_2]
        features=['llv','hlv']
    
    elif feat_set=="hlv_yolo11x8_body_faces":        
        features=['yolo11x8_resamp']
        x=np.load(f'../data/features/yolo11x8_DM_resamp.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X1=hrf_tools.apply_optimal_hrf_10hz(x,hz)

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['body'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X2 = resample(X, 750, axis=0)

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['faces'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X3 = resample(X, 750, axis=0)

        X = np.concatenate((X1, X2, X3), axis=1)

    
    elif feat_set=="llv_pbright_flo_mo_vib_sharp":
        features=['llv_pbright_flo_mo_vib_sharp']
        llv_b=np.load(f'../data/features/DM_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X1=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

        cat_DM=pd.read_csv('../data/features/DespicableMe_summary_codes_10Hz_intuitivenames.csv')
        o_f=np.asanyarray(cat_DM['motion'])[:-1]
        o_f = zscore(o_f)
        hz=o_f.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(o_f.reshape(-1, 1),hz)
        # X=hrf_tools.apply_alternative_hrf_10hz_NEW(o_f.reshape(-1, 1),hz)
        X2 = resample(X, 750, axis=0)

        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/DM_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]

        motion=np.mean(motion_features,axis=1)
        motion = resample(motion, 750, axis=0)
        scaler = StandardScaler()
        motion = scaler.fit_transform(X=motion.reshape(-1, 1),y=None)
        hz=motion.shape[0]/600 #703 seconds in friends
        X3=hrf_tools.apply_optimal_hrf_10hz_NEW(motion,hz)
        
        llv_b=np.load(f'../data/features/DM_vibrance.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X4=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
        llv_b=np.load(f'../data/features/DM_sharpness.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X5=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

        X = np.concatenate((X1, X2, X3, X4, X5), axis=1)
 
    elif feat_set=="llv_brightness4_friends_se01e02a":        
        features=['llv_brightness4_friends_se01e02a']
        llv_b=np.load(f'../data/features/friends_se01e02a_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/703 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
    elif feat_set=="llv_brightness4_friends_se01e02b":        
        features=['llv_brightness4_friends_se01e02b']
        llv_b=np.load(f'../data/features/friends_se01e02b_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/703 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
    
    elif feat_set=="llv_brightness4_center_biased":        
        features=['llv_brightness4_center_biased']
        llv_b=np.load(f'../data/features/DM_brightness_perceptual_center.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)    

    elif feat_set=="vibrance":        
        features=['vibrance']
        llv_b=np.load(f'../data/features/DM_vibrance.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
    elif feat_set=="sharpness":        
        
        features=['sharpness']
        llv_b=np.load(f'../data/features/DM_sharpness.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
    elif feat_set=="frac_high_saliency":        
        
        features=['frac_high_saliency']
        llv_b=np.load(f'../data/features/DM_frac_high_saliency.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

        
    elif feat_set=="max_saliency":        
        
        features=['max_saliency']
        llv_b=np.load(f'../data/features/DM_max_saliency.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
    elif feat_set=="llv_lab_color4":        
        
        features=['lab_color']
        lab_color=np.load(f'../data/features/DM_lab_color4.npy')
        lab_color = zscore(lab_color)
        hz=lab_color.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(lab_color,hz)    

    
    elif feat_set=="llv_contrast_brightness_motionpca5":        
        
        features=['llv_contrast_brightness']
        llv_b=np.load(f'../data/features/DM_brightness.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        llv_b=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

        llv_c=np.load(f'../data/features/DM_contrast.npy')
        llv_c = zscore(llv_c)
        hz=llv_c.shape[0]/600 #703 seconds in friends
        llv_c=hrf_tools.apply_optimal_hrf_10hz(llv_c.reshape(-1, 1),hz)
        import h5py
        
        
        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        m = resample(motion_features, 750, axis=0)
        m = scaler.fit_transform(X=m,y=None)
        n_samples=m.shape[0]
        transformer = PCA(n_components=5)
        m=transformer.fit_transform(m)
        hz=m.shape[0]/600 #703 seconds in friends
        m=hrf_tools.apply_optimal_hrf_10hz(m,hz)
        features=['llv_contrast_brightness_motionpca5']


        
        X = np.concatenate((llv_b, llv_c, m), axis=1)




    elif feat_set=="llv_aggregate2":        
        import h5py
        features=['llv_aggregate2']
        llv_b=np.load(f'../data/features/DM_contrast.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X1=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
        llv_b=np.load(f'../data/features/DM_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X2=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)    

        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        m = resample(motion_features, 750, axis=0)
        m = scaler.fit_transform(X=m,y=None)
        n_samples=m.shape[0]
        transformer = PCA(n_components=4)
        m=transformer.fit_transform(m)
        hz=m.shape[0]/600 #703 seconds in friends
        m=hrf_tools.apply_optimal_hrf_10hz(m,hz)
        
        X = np.concatenate((X1,X2, m), axis=1)
    elif feat_set=="llv_aggregate2_friends_s01e02a":        
        import h5py
        features=['llv_aggregate2_friends_s01e02a']
        llv_b=np.load(f'../data/features/friends_s01e02a_contrast.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/703 #703 seconds in friends
        X1=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
        llv_b=np.load(f'../data/features/friends_s01e02a_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/703 #703 seconds in friends
        X2=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)    

        hdf5_path = '../data/features/friends_s01e02a_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        m = resample(motion_features, 472, axis=0)
        m = scaler.fit_transform(X=m,y=None)
        n_samples=m.shape[0]
        transformer = PCA(n_components=4)
        m=transformer.fit_transform(m)
        hz=m.shape[0]/703 #703 seconds in friends
        m=hrf_tools.apply_optimal_hrf_10hz(m,hz)
        
        X = np.concatenate((X1,X2, m), axis=1)


    elif feat_set=="llv_aggregate2_friends_s01e02b":        
        import h5py
        features=['llv_aggregate2_friends_s01e02b']
        llv_b=np.load(f'../data/features/friends_s01e02b_contrast.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/703 #703 seconds in friends
        X1=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
        llv_b=np.load(f'../data/features/friends_s01e02b_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/703 #703 seconds in friends
        X2=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)    

        hdf5_path = '../data/features/friends_s01e02b_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        m = resample(motion_features, 472, axis=0)
        m = scaler.fit_transform(X=m,y=None)
        n_samples=m.shape[0]
        transformer = PCA(n_components=4)
        m=transformer.fit_transform(m)
        hz=m.shape[0]/703 #703 seconds in friends
        m=hrf_tools.apply_optimal_hrf_10hz(m,hz)
        
        X = np.concatenate((X1,X2, m), axis=1)


    
    elif feat_set=="llv_aggregate2_canon":        
        import h5py
        features=['llv_aggregate2_canon']
        llv_b=np.load(f'../data/features/DM_contrast.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X1=hrf_tools.apply_canonical_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
        llv_b=np.load(f'../data/features/DM_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X2=hrf_tools.apply_canonical_hrf_10hz(llv_b.reshape(-1, 1),hz)    

        
        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        m = resample(motion_features, 750, axis=0)
        m = scaler.fit_transform(X=m,y=None)
        n_samples=m.shape[0]
        transformer = PCA(n_components=4)
        m=transformer.fit_transform(m)
        hz=m.shape[0]/600 #703 seconds in friends
        m=hrf_tools.apply_canonical_hrf_10hz(m,hz)
        
        X = np.concatenate((X1,X2, m), axis=1)


    
    elif feat_set=="llv_aggregate":        
        import h5py
        features=['llv_brightness4']
        llv_b=np.load(f'../data/features/DM_contrast_brightness.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X1=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
        llv_b=np.load(f'../data/features/DM_brightness4.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X2=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)    

        llv_b=np.load(f'../data/features/DM_vibrance.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X3=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
        llv_b=np.load(f'../data/features/DM_sharpness.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X4=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
        llv_b=np.load(f'../data/features/DM_frac_high_saliency.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X5=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)

        llv_b=np.load(f'../data/features/DM_max_saliency.npy')
        llv_b = zscore(llv_b)
        hz=llv_b.shape[0]/600 #703 seconds in friends
        X6=hrf_tools.apply_optimal_hrf_10hz(llv_b.reshape(-1, 1),hz)
        
        lab_color=np.load(f'../data/features/DM_lab_color4.npy')
        lab_color = zscore(lab_color)
        hz=lab_color.shape[0]/600 #703 seconds in friends
        X7=hrf_tools.apply_optimal_hrf_10hz(lab_color,hz)    

        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        m = resample(motion_features, 750, axis=0)
        m = scaler.fit_transform(X=m,y=None)
        n_samples=m.shape[0]
        transformer = PCA(n_components=4)
        m=transformer.fit_transform(m)
        hz=m.shape[0]/600 #703 seconds in friends
        m=hrf_tools.apply_optimal_hrf_10hz(m,hz)

        X = np.concatenate((X1,X2,X3,X4,X5,X6,X7, m), axis=1)

    elif feat_set=="clarifai":        
        features=['clarifai']
        x=np.load(f'../data/features/DM_clarifai.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)





    elif feat_set=="yolo11x_binary":        
        features=['yolo11x_binary']
        x=np.load(f'../data/features/yolo11x_DM_binary.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)



    elif feat_set=="yolo11x_conf":        
        features=['yolo11x_conf']
        x=np.load(f'../data/features/yolo11x_DM_conf.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)
    elif feat_set=="yolo11x8_conf":        
        features=['yolo11x8_conf']
        x=np.load(f'../data/features/yolo11x8_DM_conf.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)
    elif feat_set=="yolo11x8_binary":        
        features=['yolo11x8_binary']
        x=np.load(f'../data/features/yolo11x8_DM_binary.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)




    elif feat_set=="yolo11x80_resamp":        
        features=['yolo11x80_resamp']
        x=np.load(f'../data/features/yolo11x_DM_resamp.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)

    

    elif feat_set=="yolo11x80_lpfresamp":        
        features=['yolo11x80_lpfresamp']
        x=np.load(f'../data/features/yolo11x_DM_lpfresamp.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)



    elif feat_set=="yolo11x8_resamp":        
        features=['yolo11x8_resamp']
        x=np.load(f'../data/features/yolo11x8_DM_resamp.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)


    
    elif feat_set=="yolo11x8_resamp_friends_s01e02a":        
        features=['yolo11x8_resamp_friends_s01e02a']
        x=np.load(f'../data/features/yolo11x8_friends_s01e02a_resamp.npy')
        x = zscore(x)
        hz=x.shape[0]/703 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)
    elif feat_set=="yolo11x8_resamp_friends_s01e02b":        
        features=['yolo11x8_resamp_friends_s01e02b']
        x=np.load(f'../data/features/yolo11x8_friends_s01e02b_resamp.npy')
        x = zscore(x)
        hz=x.shape[0]/703 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)

    elif feat_set=="yolo11x8_lpfresamp":        
        features=['yolo11x8_lpfresamp']
        x=np.load(f'../data/features/yolo11x8_DM_lpfresamp.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)








    
    elif feat_set=="yolov5x_class_scores80":        
        features=['yolov5x_class_scores80']
        x=np.load(f'../data/features/DM_yolov5x_class_scores80.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)


    elif feat_set=="yolov5x_class_scores80_norm":        
        features=['yolov5x_class_scores80_norm']
        x=np.load(f'../data/features/DM_yolov5x_class_scores80_norm.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)

    elif feat_set=="yolov5x_class_scores8":        
        features=['yolov5x_class_scores8']
        x=np.load(f'../data/features/DM_yolov5x_class_scores8.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)

    elif feat_set=="yolov5x_class_scores8_norm":        
        features=['yolov5x_class_scores8_norm']
        x=np.load(f'../data/features/DM_yolov5x_class_scores8_norm.npy')
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)



      
    elif feat_set=="yolov5x_class_scores80_all":        
        features=['yolov5x_class_scores80_all']
        x=np.load(f'../data/features/DM_yolov5x_class_scores80_all.npy')
        x = resample(x, 750, axis=0) #resample to 1hz for now 
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)


    elif feat_set=="yolov5x_class_scores80_norm_all":        
        features=['yolov5x_class_scores80_norm_all']
        x=np.load(f'../data/features/DM_yolov5x_class_scores80_norm_all.npy')
        x = resample(x, 750, axis=0) #resample to 1hz for now 
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)

    elif feat_set=="yolov5x_class_scores8_all":        
        features=['yolov5x_class_scores8_all']
        x=np.load(f'../data/features/DM_yolov5x_class_scores8_all.npy')
        x = resample(x, 750, axis=0) #resample to 1hz for now 

        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)

    elif feat_set=="yolov5x_class_scores8_norm_all":        
        features=['yolov5x_class_scores8_norm_all']
        x=np.load(f'../data/features/DM_yolov5x_class_scores8_norm_all.npy')
        x = resample(x, 750, axis=0) #resample to 1hz for now 
        x = zscore(x)
        hz=x.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(x,hz)




    elif feat_set=='motion_mean_pca4':
        import h5py
        from nat_asd_utils import segment_and_average
        
        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        X = segment_and_average(motion_features, 750)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        transformer = PCA(n_components=4)
        X=transformer.fit_transform(X)
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']


    elif feat_set=='motion_mean_pca4_canon':
        import h5py
        from nat_asd_utils import segment_and_average
        
        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        X = segment_and_average(motion_features, 750)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        transformer = PCA(n_components=4)
        X=transformer.fit_transform(X)
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_canonical_hrf_10hz(X,hz)
        features=['motion']


    
    elif feat_set=='motion_srp01':
        
        from sklearn.random_projection import johnson_lindenstrauss_min_dim
        from sklearn.random_projection import SparseRandomProjection
        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/DM_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]
        eps=0.1
        scaler = StandardScaler()
        X = resample(motion_features, 750, axis=0)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        n_components=johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
        if n_components < n_samples:
            srp = SparseRandomProjection(n_components=n_components,random_state=42)
            X=srp.fit_transform( X )
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']


    elif feat_set=='motion_srp05':
        
        from sklearn.random_projection import johnson_lindenstrauss_min_dim
        from sklearn.random_projection import SparseRandomProjection
        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/DM_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]
        eps=0.5
        scaler = StandardScaler()
        X = resample(motion_features, 750, axis=0)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        n_components=johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
        if n_components < n_samples:
            srp = SparseRandomProjection(n_components=n_components,random_state=42)
            X=srp.fit_transform( X )
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']
    elif feat_set=='motion_srp09':
        
        from sklearn.random_projection import johnson_lindenstrauss_min_dim
        from sklearn.random_projection import SparseRandomProjection
        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/DM_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]
        eps=0.9
        scaler = StandardScaler()
        X = resample(motion_features, 750, axis=0)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        n_components=johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
        if n_components < n_samples:
            srp = SparseRandomProjection(n_components=n_components,random_state=42)
            X=srp.fit_transform( X )
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']
    elif feat_set=='motion':
        
        # from sklearn.random_projection import johnson_lindenstrauss_min_dim
        # from sklearn.random_projection import SparseRandomProjection
        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/DM_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]
        eps=0.5
        scaler = StandardScaler()
        
        X = resample(motion_features, 750, axis=0)
        X = scaler.fit_transform(X=X,y=None)
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']



    elif feat_set=='motion_pca5':
        import h5py
        
        
        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        X = resample(motion_features, 750, axis=0)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        transformer = PCA(n_components=5)
        X=transformer.fit_transform(X)
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']
    elif feat_set=='motion_pca1':
        import h5py
        
        
        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        X = resample(motion_features, 750, axis=0)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        transformer = PCA(n_components=1)
        X=transformer.fit_transform(X)
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']

    elif feat_set=='motion_pca2':
        import h5py
        
        
        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        X = resample(motion_features, 750, axis=0)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        transformer = PCA(n_components=2)
        X=transformer.fit_transform(X)
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']

    elif feat_set=='motion_pca4':
        import h5py
        
        
        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        X = resample(motion_features, 750, axis=0)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        transformer = PCA(n_components=4)
        X=transformer.fit_transform(X)
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']


    elif feat_set=='motion_pca8':
        import h5py
        
        
        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        X = resample(motion_features, 750, axis=0)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        transformer = PCA(n_components=8)
        X=transformer.fit_transform(X)
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']

    elif feat_set=='motion_pca10':
        import h5py
        
        
        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        X = resample(motion_features, 750, axis=0)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        transformer = PCA(n_components=10)
        X=transformer.fit_transform(X)
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']

    elif feat_set=='motion_pca25':
        import h5py
        
        
        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        X = resample(motion_features, 750, axis=0)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        transformer = PCA(n_components=25)
        X=transformer.fit_transform(X)
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']

    elif feat_set=='motion_pca50':
        import h5py
        
        
        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        X = resample(motion_features, 750, axis=0)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        transformer = PCA(n_components=50)
        X=transformer.fit_transform(X)
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']

    elif feat_set=='motion_pca100':
        import h5py
        
        
        hdf5_path = '../data/features/DM_pymoten.h5'
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        X = resample(motion_features, 750, axis=0)
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        transformer = PCA(n_components=100)
        X=transformer.fit_transform(X)
        hz=X.shape[0]/600 #703 seconds in friends
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']
    
    elif feat_set=='motion_srp05_friends_s01e02a':
        
        from sklearn.random_projection import johnson_lindenstrauss_min_dim
        from sklearn.random_projection import SparseRandomProjection
        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/friends_s01e02a_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]
        eps=0.5
        scaler = StandardScaler()
        X = resample(motion_features, 471, axis=0) #resample to 471 TRs friends 750 TRs HBN
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        n_components=johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
        if n_components < n_samples:
            srp = SparseRandomProjection(n_components=n_components,random_state=42)
            X=srp.fit_transform( X )
        hz=X.shape[0]/703 #703 seconds in friends , 600 in HBN
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']
    elif feat_set=='motion_srp05_friends_s01e02b':
        
        from sklearn.random_projection import johnson_lindenstrauss_min_dim
        from sklearn.random_projection import SparseRandomProjection
        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/friends_s01e02b_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]
        eps=0.5
        scaler = StandardScaler()
        X = resample(motion_features, 471, axis=0) #resample to 471 TRs friends 750 TRs HBN
        X = scaler.fit_transform(X=X,y=None)
        n_samples=X.shape[0]
        n_components=johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
        if n_components < n_samples:
            srp = SparseRandomProjection(n_components=n_components,random_state=42)
            X=srp.fit_transform( X )
        hz=X.shape[0]/703 #703 seconds in friends , 600 in HBN
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']
    elif feat_set=='motion_friends_s01e02a':
        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/friends_s01e02a_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        X = resample(motion_features, 471, axis=0) #resample to 471 TRs friends 750 TRs HBN
        X = scaler.fit_transform(X=X,y=None)
        hz=X.shape[0]/703 #703 seconds in friends , 600 in HBN
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']
    elif feat_set=='motion_friends_s01e02b':
        
        import h5py
        # Path to the HDF5 file
        hdf5_path = '../data/features/friends_s01e02b_pymoten.h5'
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Access the dataset
            motion_features = hdf5_file['pymoten'][:]
        scaler = StandardScaler()
        X = resample(motion_features, 471, axis=0) #resample to 471 TRs friends 750 TRs HBN
        X = scaler.fit_transform(X=X,y=None)
        hz=X.shape[0]/703 #703 seconds in friends , 600 in HBN
        X=hrf_tools.apply_optimal_hrf_10hz(X,hz)
        features=['motion']

    elif feat_set=='concatspeech':
        X=[]
        feature_data=[]
        X1,features=load_features('cochresnet50pca1hrfssfirst')
        X1 = [x[:,0] for x in X1]
        X.append(X1[4])
        X.append(X1[5])
        
        X1,feats=load_features('manualhrfpca10')
        X.append(X1[3][:-1,1])
        features.append('as_embed_pca2')
        X1,feats=load_features('audioset')
        Xx = [x[:-1,:] for x in X1]
        for xx in Xx:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
        X.append(Xx[1][:,0])
        X.append(Xx[1][:,132])
        features.append('as-Speech')
        features.append('as-Music')
        X=np.asanyarray(X).T
    
    
    elif feat_set=='cochresnet50pca1_flat':
        X1,features=load_features('cochresnet50pca1hrfssfirst')
        X=np.asanyarray(X1)[:,:,0].T
        features=['cochresnet50pca1']

    elif feat_set=='testconcat':
        X=[]
        feature_data=[]
        X1,features=load_features('cochresnet50pca1hrfssfirst')
        X1 = [x[:,0] for x in X1]
        X.append(X1[4])
        X.append(X1[5])
        X1,feats=load_features('audioset')
        Xx = [x[:-1,:] for x in X1]
        for xx in Xx:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
        X.append(Xx[1][:,0])
        #X.append(Xx[1][:,132])
        features.append('as-Speech')
        #features.append('as-Music')
        X=np.asanyarray(X).T

    elif feat_set=='cochresnet50mean_input_after_preproc_hrf':
        features=['input_after_preproc']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
    elif feat_set=='cochresnet50mean_conv1_relu1_hrf':
        features=['conv1_relu1']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
    elif feat_set=='cochresnet50mean_maxpool1_hrf':
        features=['maxpool1']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
    elif feat_set=='cochresnet50mean_layer1_hrf':
        features=['layer1']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
    elif feat_set=='cochresnet50mean_layer2_hrf':
        features=['layer2']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
    elif feat_set=='cochresnet50mean_layer3_hrf':
        features=['layer3']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
    elif feat_set=='cochresnet50mean_layer4_hrf':
        features=['layer4']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
    elif feat_set=='cochresnet50mean_avgpool_hrf':
        features=['avgpool']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)

    elif feat_set=='manuallow':
        
        X=[]
        features=['rms','chroma', 'mfcc', 'mfs']
        for f in features:
            feature=np.load(f'../data/features/DM_{f}.npy')
            #print(feature.shape)
            # transformer = PCA(n_components=n_components)
            scaler = StandardScaler()
            # feature=transformer.fit_transform(feature)
            feature = scaler.fit_transform(X=feature,y=None)
            #print(feature.shape)
            feat_x = resample(feature, 750, axis=0) #resample to 1hz for now 
            X.append(feat_x)
    elif feat_set=='audioset':
        
        X=[]
        features=['as_embed', 'as_scores']
        for f in features:
            feature=np.load(f'../data/features/DM_{f}.npy')
            #print(feature.shape)
            # transformer = PCA(n_components=n_components)
            scaler = StandardScaler()
            # feature=transformer.fit_transform(feature)
            feature = scaler.fit_transform(X=feature,y=None)
            #print(feature.shape)
            feat_x = resample(feature, 750, axis=0) #resample to 1hz for now 
            X.append(feat_x)
    elif feat_set=='manualhrf':
        features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
        X=nat_asd_utils.load_audio_features_manual_hrf('DM',features)
    elif feat_set=='manualhrfpca1':
        features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
        X=nat_asd_utils.load_audio_features_manual_hrf('DM',features)
        
        X_pca=[]
        transformer = PCA(n_components=1)
        for x in X:
            X_pca.append(   transformer.fit_transform(x)   ) 
        X=X_pca
    elif feat_set=='manualhrfpca10':
        features=['chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
        X=nat_asd_utils.load_audio_features_manual_hrf('DM',features)
        
        X_pca=[]
        transformer = PCA(n_components=10)
        for x in X:
            X_pca.append(   transformer.fit_transform(x)   ) 
        X=X_pca
    elif feat_set=="cochresnet50":
        features=features_cochresnet
        X=nat_asd_utils.load_audio_features('DM',features)
    elif feat_set=="cochresnet50pca1hrfssfirst":
        
        features=features_cochresnet
        X_raw=nat_asd_utils.load_audio_features('DM',features)
        X=nat_asd_utils.standardscale(X_raw)
        X=nat_asd_utils.apply_pca(X, 1)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca20hrfssfirst":
        
        features=features_cochresnet
        X_raw=nat_asd_utils.load_audio_features('DM',features)
        X=nat_asd_utils.standardscale(X_raw)
        X=nat_asd_utils.apply_pca(X, 20)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca10hrfssfirst":
        
        features=features_cochresnet
        X_raw=nat_asd_utils.load_audio_features('DM',features)
        X=nat_asd_utils.standardscale(X_raw)
        X=nat_asd_utils.apply_pca(X, 10)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)            
    elif feat_set=="cochresnet50pca100hrfssfirst":
        
        features=features_cochresnet
        X_raw=nat_asd_utils.load_audio_features('DM',features)
        X=nat_asd_utils.standardscale(X_raw)
        X=nat_asd_utils.apply_pca(X, 100)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    
    
    elif feat_set=="cochresnet50srp05hrfssfirst":
        
        features=features_cochresnet
        X_raw=nat_asd_utils.load_audio_features('DM',features)
        X=nat_asd_utils.standardscale(X_raw)
        X=nat_asd_utils.apply_srp(X,0.5)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50srp01hrfssfirst":
        
        features=features_cochresnet
        X_raw=nat_asd_utils.load_audio_features('DM',features)
        X=nat_asd_utils.standardscale(X_raw)
        X=nat_asd_utils.apply_srp(X,0.1)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)

    elif feat_set=="cochresnet50pca1hrf":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="resnet50pca1hrf":
        features=features_resnet
        feature_filename='DM_resnet50_activations-PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="resnet50pca5hrf":
        features=features_resnet
        feature_filename='DM_resnet50_activations-PCA-5.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="resnet50pca10hrf":
        features=features_resnet
        feature_filename='DM_resnet50_activations-PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz) 
    elif feat_set=="resnet50pca50hrf":
        features=features_resnet
        feature_filename='DM_resnet50_activations-PCA-50.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz) 
    elif feat_set=="resnet50pca100hrf":
        features=features_resnet
        feature_filename='DM_resnet50_activations-PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz) 
    elif feat_set=="resnet50pca200hrf":
        features=features_resnet
        feature_filename='DM_resnet50_activations-PCA-200.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz) 
    elif feat_set=="resnet50pca500hrf":
        features=features_resnet
        feature_filename='DM_resnet50_activations-PCA-500.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)

    elif feat_set=="cochresnet50pca1hrffriends_s01e02a":
        features=features_cochresnet
        feature_filename='friends_s01e02a_cochresnet50_activations-mean_PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca10hrffriends_s01e02a":
        features=features_cochresnet
        feature_filename='friends_s01e02a_cochresnet50_activations-mean_PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca100hrffriends_s01e02a":
        features=features_cochresnet
        feature_filename='friends_s01e02a_cochresnet50_activations-mean_PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca1hrffriends_s01e02b":
        features=features_cochresnet
        feature_filename='friends_s01e02b_cochresnet50_activations-mean_PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca10hrffriends_s01e02b":
        features=features_cochresnet
        feature_filename='friends_s01e02b_cochresnet50_activations-mean_PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca100hrffriends_s01e02b":
        features=features_cochresnet
        feature_filename='friends_s01e02b_cochresnet50_activations-mean_PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)

    elif feat_set=="cochresnet50srp01hrffriends_s01e02a":
        features=features_cochresnet
        X=nat_asd_utils.load_audio_features_SRP('friends_s01e02a',features,0.1)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50srp01hrffriends_s01e02b":
        features=features_cochresnet
        X=nat_asd_utils.load_audio_features_SRP('friends_s01e02b',features,0.1)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)    
    elif feat_set=="cochresnet50srp05hrffriends_s01e02a":
        features=features_cochresnet
        X=nat_asd_utils.load_audio_features_SRP('friends_s01e02a',features,0.5)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50srp05hrffriends_s01e02b":
        features=features_cochresnet
        X=nat_asd_utils.load_audio_features_SRP('friends_s01e02b',features,0.5)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)    


    elif feat_set=='audioset_friends_s01e02a':
        
        X=[]
        features=['as_embed', 'as_scores']
        for f in features:
            feature=np.load(f'../data/features/s1friends_s01e02a_{f}.npy')
            scaler = StandardScaler()
            feature = scaler.fit_transform(X=feature,y=None)
            X.append(feature)
        for xx in X:
            hz=xx.shape[0]/703 #703 seconds in friends
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)    
        X_out=[]
        for xx in X:
            xx = resample(xx, 471, axis=0) #resample to 471 TRs
            X_out.append(xx)
        X=X_out
    elif feat_set=='audioset_friends_s01e02b':
        
        X=[]
        features=['as_embed', 'as_scores']
        for f in features:
            feature=np.load(f'../data/features/s1friends_s01e02b_{f}.npy')
            scaler = StandardScaler()
            feature = scaler.fit_transform(X=feature,y=None)
            X.append(feature)
        for xx in X:
            hz=xx.shape[0]/703 #703 seconds in friends
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)    
        X_out=[]
        for xx in X:
            xx = resample(xx, 471, axis=0) #resample to 471 TRs
            X_out.append(xx)
        X=X_out
    
    elif feat_set=='audioset_srp05_friends_s01e02a':
        
        X=[]
        features=['as_embed', 'as_scores']
        for f in features:
            feature=np.load(f'../data/features/s1friends_s01e02a_{f}.npy')
            scaler = StandardScaler()
            feature = scaler.fit_transform(X=feature,y=None)
            X.append(feature)
        X=nat_asd_utils.apply_srp(X,0.5)
        for xx in X:
            hz=xx.shape[0]/703 #703 seconds in friends
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)    
        X_out=[]
        for xx in X:
            xx = resample(xx, 471, axis=0) #resample to 471 TRs
            X_out.append(xx)
        X=X_out
    elif feat_set=='audioset_srp05_friends_s01e02b':
        
        X=[]
        features=['as_embed', 'as_scores']
        for f in features:
            feature=np.load(f'../data/features/s1friends_s01e02b_{f}.npy')
            scaler = StandardScaler()
            feature = scaler.fit_transform(X=feature,y=None)
            X.append(feature)
        X=nat_asd_utils.apply_srp(X,0.5)
        for xx in X:
            hz=xx.shape[0]/703 #703 seconds in friends
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)    
        X_out=[]
        for xx in X:
            xx = resample(xx, 471, axis=0) #resample to 471 TRs
            X_out.append(xx)
        X=X_out

    elif feat_set=='manualhrf_srp05_friends_s01e02a':
        
        X=[]
        features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
        for f in features:
            feature=np.load(f'../data/features/friends_s01e02a_{f}.npy')
            scaler = StandardScaler()
            feature = scaler.fit_transform(X=feature,y=None)
            X.append(feature)
        X=nat_asd_utils.apply_srp(X,0.5)
        for xx in X:
            hz=xx.shape[0]/703 #703 seconds in friends
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)    
        X_out=[]
        for xx in X:
            xx = resample(xx, 471, axis=0) #resample to 471 TRs
            X_out.append(xx)
        X=X_out
    elif feat_set=='manualhrf_srp05_friends_s01e02b':
        
        X=[]
        features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
        for f in features:
            feature=np.load(f'../data/features/friends_s01e02b_{f}.npy')
            scaler = StandardScaler()
            feature = scaler.fit_transform(X=feature,y=None)
            X.append(feature)
        X=nat_asd_utils.apply_srp(X,0.5)
        for xx in X:
            hz=xx.shape[0]/703 #703 seconds in friends
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)    
        X_out=[]
        for xx in X:
            xx = resample(xx, 471, axis=0) #resample to 471 TRs
            X_out.append(xx)
        X=X_out


    elif feat_set=='manualhrf_srp01_friends_s01e02a':
        
        X=[]
        features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
        for f in features:
            feature=np.load(f'../data/features/friends_s01e02a_{f}.npy')
            scaler = StandardScaler()
            feature = scaler.fit_transform(X=feature,y=None)
            X.append(feature)
        X=nat_asd_utils.apply_srp(X,0.1)
        for xx in X:
            hz=xx.shape[0]/703 #703 seconds in friends
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)    
        X_out=[]
        for xx in X:
            xx = resample(xx, 471, axis=0) #resample to 471 TRs
            X_out.append(xx)
        X=X_out
    elif feat_set=='manualhrf_srp01_friends_s01e02b':
        
        X=[]
        features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
        for f in features:
            feature=np.load(f'../data/features/friends_s01e02b_{f}.npy')
            scaler = StandardScaler()
            feature = scaler.fit_transform(X=feature,y=None)
            X.append(feature)
        X=nat_asd_utils.apply_srp(X,0.1)
        for xx in X:
            hz=xx.shape[0]/703 #703 seconds in friends
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)    
        X_out=[]
        for xx in X:
            xx = resample(xx, 471, axis=0) #resample to 471 TRs
            X_out.append(xx)
        X=X_out

    
    elif feat_set=="video_resnet50pca1hrf":
        features=features_resnet
        feature_filename='DM_videos_resnet50-PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="video_resnet50pca10hrf":
        features=features_resnet
        feature_filename='DM_videos_resnet50-PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="video_resnet50pca100hrf":
        features=features_resnet
        feature_filename='DM_videos_resnet50-PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="video_slowfastr50pca1hrf":
        features=features_slowfast
        feature_filename='DM_videos_slowfast_r50-PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="video_slowfastr50pca10hrf":
        features=features_slowfast
        feature_filename='DM_videos_slowfast_r50-PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="video_slowfastr50pca100hrf":
        features=features_slowfast
        feature_filename='DM_videos_slowfast_r50-PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca1":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca200":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-200.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca5":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-5.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca10":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca50":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-50.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca100":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca1friends_s01e02a":
        features=features_cochresnet
        feature_filename='friends_s01e02a_cochresnet50_activations-mean_PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca10friends_s01e02a":
        features=features_cochresnet
        feature_filename='friends_s01e02a_cochresnet50_activations-mean_PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca100friends_s01e02a":
        features=features_cochresnet
        feature_filename='friends_s01e02a_cochresnet50_activations-mean_PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca1friends_s01e02b":
        features=features_cochresnet
        feature_filename='friends_s01e02b_cochresnet50_activations-mean_PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca10friends_s01e02b":
        features=features_cochresnet
        feature_filename='friends_s01e02b_cochresnet50_activations-mean_PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca100friends_s01e02b":
        features=features_cochresnet
        feature_filename='friends_s01e02b_cochresnet50_activations-mean_PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull1":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull200":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-200.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull5":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-5.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull10":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull50":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-50.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull100":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcac2":
        features=features_cochresnet
        X=nat_asd_utils.load_audio_features_PCAc2('DM',features)
    elif feat_set=="cochresnet50PCAlocal1":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal10":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal1mean":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-1_mean.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal10mean":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-10_mean.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal1rev":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-1_rev.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal5rev":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-5_rev.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="both_hrf":
        X=nat_asd_utils.load_both_features_hrf('DM')
        features=['input_after_preproc',
                  'conv1_relu1',
                  'layer1',
                  'layer2',
                  'layer3',
                  'layer4',
                  'avgpool',
                  'as_embed', 
                  'as_scores']
    return(X,features)

def plot_violins(r2s, stacked_r2s, S_average, features, output_name):
    plot_data=np.concatenate((r2s, stacked_r2s.reshape(1, -1)), axis=0).T
    fig, axs = plt.subplots(2, figsize=(7, 10))
    axs = axs.flatten()
    
    #plt.figure(figsize=(6,4))
    sns.violinplot(data=plot_data,ax=axs[0])
    #    plt.xticks(np.arange(24), ['Subject'+str(k+1) for k in range(24)])
    #plt.title(f'R2 for each feature space')
    axs[0].set_title(f'R2 for each feature space')
    
    #features=all_layers
    #features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
    
    # Add 'stacked' to the end of your features list
    labels = features + ['stacked']
    
    # Set the xticklabels of your plot
    axs[0].set_xticks(range(len(labels)))
    axs[0].set_xticklabels(labels,rotation=45)
    
    plt.figure(figsize=(6,4))
    sns.violinplot(data=S_average,ax=axs[1])
    #    plt.xticks(np.arange(24), ['Subject'+str(k+1) for k in range(24)])
    #plt.title(f'Violin plot of the subjects by delay for brain region {parcel_names[j]}')
    labels = features
    axs[1].set_title(f'Stacking weights')
    
    # Set the xticklabels of your plot
    axs[1].set_xticks(range(len(labels)))
    axs[1].set_xticklabels(labels,rotation=45)

    fig.savefig(f'../plots/pilots/{output_name}.png')

def load_sub_brain(sub,delay,atlas_indices_indices):    
    #im_file = f'/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/xcp_d_0.7.1/sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'
    #im_file = f'/om2/scratch/tmp/jsmentch/HBN/sub-{sub}/data/derivatives/xcp_d/sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'
    im_file = f'/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/xcp_d_0.7.5_cifti/sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'
    img = nb.load(im_file)
    img_y = img.get_fdata()
    Y=img_y[delay:,atlas_indices_indices]
    print(f'loaded brain data')
    return(Y)


def load_sub_brain_friends(sub,task,delay,atlas_indices_indices):        
    # old -- these don't exist anymore??
    # fmriprep_folder='/nese/mit/group/sig/projects/cneuromod/friends/postproc_fmriprep'
    # pattern=f'{fmriprep_folder}/sub-{sub}/sub-{sub}_ses-*_task-{task}_space-fsLR_den-91k_bold_smoothed.dtseries.nii'

    fmriprep_folder='/nese/mit/group/sig/projects/cneuromod/postproc_fmriprep/friends'
    pattern=f'{fmriprep_folder}/sub-{sub}/sub-{sub}_ses-*_task-{task}_space-fsLR_den-91k_bold_cleaned_smoothed.dtseries.nii'
    
    im_file = glob.glob(pattern)[0]
    img = nb.load(im_file)
    img_y = img.get_fdata()
    Y=img_y[delay:,atlas_indices_indices]
    # print(f'loaded brain data')
    return(Y)
    
def load_sub_brain_all(sub,delay):    
    #im_file = f'/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/xcp_d_0.7.1/sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'
    #im_file = f'/om2/scratch/tmp/jsmentch/HBN/sub-{sub}/data/derivatives/xcp_d/sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'
    im_file = f'/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/xcp_d_0.7.5_cifti/sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'
    img = nb.load(im_file)
    img_y = img.get_fdata()
    Y=img_y[delay:,:]
    print(f'loaded brain data')
    return(Y)

def extract_parcels(parcels):
    atlas,atlas_data=nat_asd_utils.load_glasser()
    atlas_indices,indices,parcel_names=nat_asd_utils.get_parcel_indices(atlas,parcels)
    atlas_indices_indices = np.where(np.isin(atlas_data, atlas_indices))[0]
    #print(f'loaded parcels {parcels}')
    return(atlas_indices_indices)

def select_parcels(parcel_selection):
    #parcels: auditory, visual, audiovideo, all, custom
    if parcel_selection == 'auditory':
        parcels=[
        'A1',
        'LBelt',
        'MBelt',
        'PBelt',
        'A4',
        'TA2',
        'A5']
    elif parcel_selection == 'a4a5':
        parcels=[
        'A4',
        'A5']
    elif parcel_selection == 'A1':
        parcels=[
        'A1']
    elif parcel_selection == 'visual':
        parcels=[
                'V1',
                'V2',
                'V3',
                'V4',
                'MT',
                'MST',
                'V4t',
                'FST',
                'FFC',
                'V8',
                'PIT',
                'VVC',
                'VMV1',
                'VMV2',
                'VMV3',
                'V3A',
                'V3B',
                'V6',
                'V6A',
                'V7']
    elif parcel_selection == 'ffcvvc':
        parcels=[
                'FFC',
                'VVC']
    elif parcel_selection == 'earlyvisual':
        parcels=[
                'V1',
                'V2',
                'V3',
                'V4',
                'MT']
    elif parcel_selection == 'V1':
        parcels=[
                'V1']
    elif parcel_selection == 'MT':
        parcels=[
                'MT']
    elif parcel_selection == 'ventralvisual':
        parcels=[
                'FFC',
                'V8',
                'PIT',
                'VVC',
                'VMV1',
                'VMV2',
                'VMV3']
    elif parcel_selection == 'audiovisual':
        parcels=[
                'IPS1',
                'IFSa',
                'IFSp',
                'IFJa',
                'IFJp',
                'FEF',
                'STSvp',
                'STSdp',
                'STSva',
                'STSda',
                'STGa',
                'STV',
                'TPOJ1',
                'TPOJ2',
                'TPOJ3']
    elif parcel_selection == 'all_select':
        parcels=[
                'V1',
                'V2',
                'V3',
                'V4',
                'MT',
                'MST',
                'V4t',
                'FST',
                'FFC',
                'V8',
                'PIT',
                'VVC',
                'VMV1',
                'VMV2',
                'VMV3',
                'V3A',
                'V3B',
                'V6',
                'V6A',
                'V7',
                'IPS1',
                'IFSa',
                'IFSp',
                'IFJa',
                'IFJp',
                'FEF',
                'STSvp',
                'STSdp',
                'STSva',
                'STSda',
                'STGa',
                'STV',
                'TPOJ1',
                'TPOJ2',
                'TPOJ3',
                'A1',
                'LBelt',
                'MBelt',
                'PBelt',
                'A4',
                'TA2',
                'A5']
    elif parcel_selection == 'av':
        parcels=[
                'V1',
                'V2',
                'V3',
                'V4',
                'MT',
                'MST',
                'FFC',
                'V8',
                'PIT',
                'VVC',
                'STSvp',
                'STSdp',
                'STSva',
                'STSda',
                'STGa',
                'TPOJ1',
                'TPOJ2',
                'TPOJ3',
                'A1',
                'LBelt',
                'MBelt',
                'PBelt',
                'A4',
                'A5']
    else:
        parcels=parcel_selection
    return(parcels)

if __name__ == "__main__":
    main()
