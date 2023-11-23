from scipy.io import loadmat
from utils.utils import *
import scipy.io as sio
from os.path import join
from hyperopt import fmin, tpe, hp
import numpy as np

"""
    This section is about the fusion module based on Bayesian optimization
"""

result_path = "./exp_fusion"

# Load data
A = loadmat('./data/rawdata49.mat')
A = A['rawdata'][:]
org = ssos(ifft2n(A, [0, 1]))
print('org-shape:', A.shape)

# Load mask
B = loadmat('./mask/mask_1DRU_320x320_R3.mat')
mask = B['mask']
mask1 = np.expand_dims(mask, -1)
mask1 = np.tile(mask1, [1, 1, 15])

# Load the result of network1 with no DC
C = loadmat('./exp_fusion/IUNN_knee_rawdata49_mask_1DRU_R3_nodc.mat')['recon'][:]
pre1 = ssos(ifft2n(C, [0, 1]))

# Load the result of network1 with DC
C_DC = loadmat('./exp_fusion/IUNN_knee_rawdata49_mask_1DRU_R3_dc.mat')['recon'][:]
pre1_DC = ssos(ifft2n(C_DC,[0,1]))

# Load the result of network2 with no DC
D = loadmat('./exp_fusion/KUSNNS_knee_rawdata49_mask_1DRU_R3_nodc.mat')['recon'][:]
pre2 = ssos(ifft2n(D, [0, 1]))

# Load the result of network2 with DC
D_DC = loadmat('./exp_fusion/KUSNNS_knee_rawdata49_mask_1DRU_R3_dc.mat')['recon'][:]
pre2_DC = ssos(ifft2n(D_DC, [0, 1]))

# Undersample all the data.
c = ssos(ifft2n(C*mask1, [0, 1]))
d = ssos(ifft2n(D*mask1, [0, 1]))
a = ssos(ifft2n(A*mask1, [0, 1]))

# Define the names and number of parameters
def objective1(params):
    K, I = params
    z = np.mean((c*K + d*I - a)**2)
    return z

# Calculate the fusion ratio of three times
for i in range(3):
 
    space = [hp.uniform('K', 0, 1), hp.uniform('I', 0, 1)]
    best = fmin(objective1, space, algo=tpe.suggest, max_evals=1000)
    print(best)

    fusion_mat = best['K']*pre1_DC + best['I']*pre2_DC
    sio.savemat(join(result_path, 'DFUSNN_knee_rawdata49_1DRU_R3-%d.mat' % (i+1)), {'recon': fusion_mat})


# Load fusion result
recon1 = loadmat('./exp_fusion/DFUSNN_knee_rawdata49_1DRU_R3-1.mat')['recon'][:]

recon2 = loadmat('./exp_fusion/DFUSNN_knee_rawdata49_1DRU_R3-2.mat')['recon'][:]

recon3 = loadmat('./exp_fusion/DFUSNN_knee_rawdata49_1DRU_R3-3.mat')['recon'][:]

# Take the average of the fusion results.
RECON = (recon1+recon2+recon3)/3

# Save the final fusion result
sio.savemat(join(result_path, 'DFUSNN_knee_rawdata49_1DRU_R3-RECON.mat'), {'recon': RECON})



