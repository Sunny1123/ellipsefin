import pickle
from pathlib import Path
import numpy as np
import sys
import time


def ReadFile(filename):
    with open(filename, 'rb') as file:
        res = pickle.load(file)
        return res


def WriteFile(filename, obj):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def FileExists(filename):
    return Path(filename).is_file()


def rgb2grey(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def pad(x, max):
    x = abs(x)
    x = min(x, max - abs(max-x))
    return(x)
    
class Bar(object):
    def __init__(self, max, status="Working", style=["\u2588", "percent"]):
        self.max = max
        self.style = style
        self.curr = 0
        self.string = status
        self.start = time.time()

    def next(self):
        self.curr += 1
        ratio = (self.curr/self.max)*30
        string = f" [{self.string}] "
        key = self.style[0]
        string += "|"
        for i in range(30):
            if i < ratio:
                string += key
            else:
                string += "_"
        if self.style[1] == "ratio":
            string += f"|     {self.curr}/{self.max} "
        elif self.style[1] == "percent":
            string += f"|     {int(self.curr/self.max *100)} % "
        sys.stdout.write("%s\r" % string)
        sys.stdout.flush()

    def __del__(self):
        self.curr = 0
        print(f"\n\n|===============Finished {self.string}================|\n")
        print(f"{self.string} took {int(time.time()-self.start)} seconds\n")


def Add_Noise(image, lvl=10,mode = 1):
    if mode ==1:
        # rows, cols = image.shape
        # noise = np.random.randn(rows, cols)
        # res = image + noise*(np.max(image)-np.min(image))*lvl
        rows, cols = image.shape
        noise = np.random.randn(rows,cols)
        res = image+lvl*noise
    elif mode==2:
        rows, cols = image.shape
        noise = np.random.uniform(-3**0.5*lvl,3**0.5*lvl,(rows,cols))
        res = image+noise
    elif mode==3:
        rows, cols = image.shape
        noise = 0.5**0.5*lvl*np.random.laplace(size=(rows,cols))
        res = image+noise
    # res = res - min(np.min(res), 0)
    # if np.max(res) > 5:
    #     res = res / max(np.max(res), 255)
    # else:
    #     res = res / max(np.max(res), 1)
    # if np.max(res) > 10 :
    #     res = res/255
    # for i in range(rows):
    #     for j in range(cols):
    #         if res[i,j]<0:
    #             res[i,j]=-res[i,j]
    #         elif res[i,j]>1:
    #             res[i,j] = 2 - res[i,j]
    return res

def edgemse(img):
    from denoise.utils import pad
    rows, cols = img.shape
    img =  img.astype(float)
    sum =0
    for i in range(rows):
        for j in range(cols):
            sum = sum + (img[pad(i-1,rows-1),j]-img[pad(i+1,rows-1),j])**2
            sum = sum + (img[pad(i-1,rows-1),pad(j-1,cols-1)]-img[pad(i+1,rows-1),pad(j+1,cols-1)])**2
            sum = sum + (img[i,pad(j-1,cols-1)]-img[i,pad(j+1,cols-1)])**2
            sum = sum + (img[pad(i-1,rows-1),pad(j+1,cols-1)]-img[pad(i+1,rows-1),pad(j-1,cols-1)])**2
    return (sum/(4*rows*cols))**(0.5)
def summary(image, estimates):
    rows, cols = image.shape
    rmse = (np.sum((image-estimates)**2)/(rows*cols))**(0.5)
#   print(f"debug {np.max(self.image)} {((self.image-self.estimates)**2).shape} {np.std(self.estimates)}")
    res = f"image std = {np.std(image)} rmse = {rmse} \n"
    emse_orig= edgemse(image)
    emse_estim = edgemse(estimates)
    emse_diff = emse_orig-emse_estim
    res =res + f"image 1 \t{emse_orig}\n image 2 \t{emse_estim}\n difference\t{emse_diff}\n"
    return res

import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# -----DEBUG--------
