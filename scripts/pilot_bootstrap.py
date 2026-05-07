import pandas as pd
import nibabel as nb
import os
import numpy as np
import glob
import h5py
import hcp_utils as hcp
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA

import numpy as np
from naturalistic_encoding.stacking_fmri import stacking_CV_fmri, stacking_fmri
from naturalistic_encoding.ridge_tools import R2
import matplotlib.pyplot as plt
import seaborn as sns

import time
import naturalistic_encoding.nat_asd_utils as nat_asd_utils

import sys

start_time = time.time()

sub=str(sys.argv[1])
print(f'running subject {sub}')

atlas,atlas_data=nat_asd_utils.load_glasser()

# sub = "sub-01"  # example only; use your anonymous participant id

#parcel='A1'

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

atlas_indices,indices,parcel_names=nat_asd_utils.get_parcel_indices(atlas,parcels)

atlas_indices_indices = np.where(np.isin(atlas_data, atlas_indices))[0]

#atlas_indices_indices=np.where( (atlas_data==204) | (atlas_data==24) )[0]

print(f'loaded parcels {parcels}')

n_components=1
delay=6

im_file = f'/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/xcp_d_0.7.1/sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'
img = nb.load(im_file)
img_y = img.get_fdata()
Y=img_y[delay:,atlas_indices_indices]

print(f'loaded brain data')


all_layers=['input_after_preproc',
 'conv1_relu1',
 'maxpool1',
 'layer1',
 'layer2',
 'layer3',
 'layer4',
 'avgpool']

# all_layers=['input_after_preproc',
#  'conv1_relu1',
#  'maxpool1']
X=nat_asd_utils.load_audio_features_PCA('DM',delay,all_layers,n_components)
#X=load_audio_features('DM',delay,all_layers)

X = [array[:Y.shape[0], :] for array in X]
Y= Y[:X[0].shape[0],:]

for bootstrap in np.arange(100):
        
    #randomly permute the order of X
    for i in range(len(X)):
        np.random.shuffle(X[i])
    
    print(f'begin regression')
    
    
    r2s, stacked_r2s, r2s_weighted, _, _, S_average = stacking_CV_fmri(Y, X, method = 'cross_val_ridge',n_folds = 5,score_f=R2)
    
    elapsed_time=time.time() - start_time
    print(elapsed_time)
    
    print(f'saving results')
    
    np.savez(f'../pilot_results/feat-audio_sub-{sub}_ROI-all_PCA-{n_components}_delay-{delay}_bootstrap-{bootstrap}', r2s=r2s, stacked_r2s=stacked_r2s, r2s_weighted=r2s_weighted, S_average=S_average, elapsed_time=elapsed_time)