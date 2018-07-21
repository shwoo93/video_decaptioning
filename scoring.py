import os
import matplotlib.pyplot as plt
import numpy as np
import json
from skimage.measure import compare_ssim as ssim,compare_mse,compare_psnr
import numpy as np
import argparse
import glob
#import data_manager

def resize(inp,sz = 128):
    return imresize(inp,(sz,sz))[:,:,:3]

def PSNR(ximg,yimg):
    return compare_psnr(ximg,yimg,data_range=1)

def DSSIM(y,t,value_range=1):
    n_dims, dims = len(y.shape),y.shape
    sample_dims = (n_dims-3,n_dims-2,n_dims-1)
    #Reorder dimensions
    d_order = [i for i in range(n_dims) if i not in sample_dims] + [sample_dims[0],sample_dims[1],sample_dims[2]]
    y,t = np.transpose(y,axes = tuple(d_order)),np.transpose(t,axes=tuple(d_order))
    u_shape = y.shape[:-3]
    #Flatten non-image dimensions,calculate DSSIM
    y,t = np.reshape(y,(-1,)+y.shape[-3:]),np.reshape(t,(-1,) + y.shape[-3:])
    try:
        dssim = [(1 - ssim(
            ty, tt, gaussian_weights=True, data_range=value_range, multichannel=True
                ))/2 for ty, tt in zip(y,t)]
    except ValueError:
        #WinSize too small
        dssim = [(1 - ssim(
            ty, tt, gaussian_weights=True, data_range=value_range, multichannel=True, win_size=3
                ))/2 for ty, tt in zip(y,t)]
    return dssim

def MSE(x,y):
	return compare_mse(x,y)

'''def Evaluate(files_gt, files_pred, methods = [PSNR,MSE,DSSIM]):
    score = {}
    for meth in methods:
        name = meth.__name__
        results = []
        
        for pred, real in zip(files_pred, files_gt):
            Ypred = data_manager.getAllFrames(pred)
            Yreal = data_manager.getAllFrames(real)
            if Ypred.shape[0] == Yreal.shape[0]:
                res = 0.
                for frame in range(Ypred.shape[0]):
                    res += meth(Ypred[frame], Yreal[frame])
                res /= float(Ypred.shape[0])
                results.append(res)

        mres = np.mean(results)
        print(name+": "+str(mres)+" Std: "+str(np.std(results)))
        score[name]=mres
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('DataFolder')
    parser.add_argument('ResultsFolder')
    # usage: python -i scoring.py '../data/dataset-mp4/dev' 'outputs/B2/'

    args = parser.parse_args()
    Yreal = sorted(glob.glob(os.path.join(args.DataFolder,"Y","*.mp4")))
    Ypred = sorted(glob.glob(os.path.join(args.ResultsFolder,"*.mp4")))
    
    args = parser.parse_args()
    Ypred = sorted(glob.glob(os.path.join(args.ResultsFolder,"*.mp4")))
    Yreal = sorted(glob.glob(os.path.join(args.DataFolder,"Y","*.mp4")))

    if len(Ypred) != len(Yreal): #something looks anormal, need to get intersection of both set
        Yreal2 = []
        for path in Ypred:
            p = os.path.basename(path).replace('.mp4.mp4','.mp4')
            Yreal2.append(glob.glob(os.path.join(args.DataFolder,"Y","*"+p+"*"))[0])
        Yreal = Yreal2

    results = Evaluate(Yreal,Ypred, methods=[MSE])
    json.dump(results, open('results.json','w')) 
'''