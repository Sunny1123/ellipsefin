from denoise.lm import Kernel, LocalKfit, Distance, Epanechnikov
import numpy as np
from denoise.utils import Bar
import cv2
import matplotlib.pyplot as plt
import matplotlib as mp
from joblib import Parallel, delayed, cpu_count
import time, tqdm
from tqdm_joblib import tqdm_joblib
from denoise.utils import pad
from scipy.stats import chi2
from math import floor, ceil
# --------------------------------------------
#
# 	Name: Grads (class)
#
# 	Desc:
#
# 	Comments:
#
# --------------------------------------------


class Grads(object):

    def Grad(self):
        image = self.image
        Kernel = self.Kernel_grad

        def res(pos):
            k = Kernel.eval(pos)
            return(LocalKfit(image, pos, k, 2))
        return res

    def eval(self):
        rows = self.rows
        cols = self.cols
        func = self.Grad
        def func_para(i,j):
            return func([i,j])[1:3]
        t1=time.time()
        n_jobs=cpu_count()-1
        batch_size= int(rows*cols/n_jobs)
        with tqdm_joblib(desc="Grad Calculation",total = rows*cols) as progress_bar:
            res = Parallel(n_jobs=n_jobs,prefer="processes",batch_size=batch_size)(delayed(func_para)(i,j) for i in range(rows) for j in range(cols))
        res=np.array(res,dtype='float64').reshape((rows,cols,2))
        print(f"Calculating gradients to {time.time()-t1} seconds")
        return res

    def reset_Grads(self):
        self.Kernel_grad = Kernel(
            self.rows, self.cols, self.band_grad, self.distfunc_grad)
        self.Grad = Grads.Grad(self)
        self.gradmat = Grads.eval(self)

    __defined = False

    def __init__(self, image, band, distfunc=Epanechnikov):
        self.image = image
        self.distfunc_grad = distfunc
        self.rows = image.shape[0]
        self.cols = image.shape[1]
        self.band_grad = band
        self._observed = ["band_grad", "image", "distfunc_grad"]
        Grads.reset_Grads(self)
        self.__defined = True

    def __setattr__(self, name, value):
        self._dist__[name] = value
        if self.__defined and name in self._observed:
            Grads.reset_Grads(self)


# ----------------------------------------------------------------
#
# 	Name: Edges (class)
#
# 	Desc: Edge class made from image bandwidth and cutoff value
#
# 	Comments: None
#
# ----------------------------------------------------------------

class Edges(Grads):

    __defined = False

    # def reset_Edges(self):
    #     # grad = self.gradmat
    #     # edgesum = np.sum(grad[:,:,0:2]**2, axis=2)
    #     # cut = np.quantile(edgesum, self.cutoff)
    #     # self.edge = edgesum > cut
    #     grad = self.gradmat
    #     img=(self.image)
    #     edgesum = np.sum(grad[:,:,0:2]**2, axis=2)
    #     cut2 = (np.quantile(edgesum, self.cutoff))**(0.5)
    #     cut1=2*cut2
    #     mp.image.imsave("temp.png",img)
    #     img=cv2.imread("temp.png",0)
    #     # plt.imshow(img,cmap="gray")
    #     # plt.show()
    #     print(f"{cut2} ------ {cut1} ------- {img.dtype} --- {np.max(img)}")
    #     self.edge = cv2.Canny(img,cut1,cut2)>0
    #     # plt.imshow(self.edge,cmap ="Greys")
    #     # plt.show()

    def reset_Edges(self):
        grad = self.gradmat
        img = self.image
        band1 = self.band_grad
        band1 = min(band1) #works for square images only
        cut = self.cutoff
        edgemat = np.zeros(img.shape)
        rows, cols = img.shape
        var =0
        for i in range(1,rows-1):
            for j in range(1,cols-1):
                avg = img[i+1,j+1]+img[i+1,j-1]+img[i-1,j+1]+img[i-1,j-1]+img[i+1,j]+img[i-1,j]+img[i,j+1]+img[i,j-1] + img[i,j]
                avg = avg/9
                var = var + (img[i,j] -avg)**2
        var = var/((rows-2)*(cols-2))
        #print(var)
        cut = chi2.ppf(cut,2)*var/(band1*(band1+1)*(2*band1+1)**2/6)
        for i in range(rows):
            for j in range(cols):
                g = grad[i,j,:]
                if g[0]==0:
                    theta = 999999
                else:
                    theta = g[1]/g[0]
                if abs(theta) <= 0.5:
                    beta1 =  grad[pad(i+1,rows -1),pad(j,cols -1)]
                    beta2 =  grad[pad(i-1,rows -1),pad(j,cols -1)]
                elif abs(theta) >= 2:
                    beta1 =  grad[pad(i,rows -1),pad(j+1,cols -1)]
                    beta2 =  grad[pad(i,rows -1),pad(j-1,cols -1)]
                elif theta > 0.5 and theta < 2 :
                    beta1 =  grad[pad(i+1,rows -1),pad(j+1,cols -1)]
                    beta2 =  grad[pad(i-1,rows -1),pad(j-1,cols -1)]
                elif theta > -2 and theta < -0.5:
                    beta1 =  grad[pad(i+1,rows -1),pad(j-1,cols -1)]
                    beta2 =  grad[pad(i-1,rows -1),pad(j+1,cols -1)]
                delta1 = sum((beta1-g)**2)
                delta2 = sum((beta2--g)**2)
                delta = min(delta1,delta2)
                edgemat[i,j] = delta > cut
        #self.edge =  edgemat
        edgemat = Edges.thin_edges(edgemat,grad,band1)
        self.edge = Edges.remove_scattered(edgemat,[5*band1,5*band1])
        self.var = var
    def remove_scattered(edgemat, band=[2,2] ):
        rows, cols = edgemat.shape
        mat = np.copy(edgemat)
        ker = Kernel(rows,cols, band , Distance)
        c = (min(band))
        print(c)
        for i in range(rows):
            for j in range(cols):
                k = ker.eval([i,j])
                def f(a, max):
                    for i in range(len(a)):
                        a[i] = abs(a[i])
                        a[i] = min(a[i], max - abs(max - a[i]))
                    return a
                k[0] = f(k[0],rows -1)
                k[1]= f(k[1],cols - 1)
                s = np.sum(mat[k[0],k[1]])
                mat[i,j] = (s > c)*mat[i,j]
        return mat

    def thin_edges(edgemat,grad,band):
        thinedgemat = np.copy(edgemat)
        rows = edgemat.shape[0]
        band1 = 2*band+1
        band1 = (3*band1 + 1)/2
        for i in range(rows):
            rowedges = np.nonzero(edgemat[:,i])[0]
            if len(rowedges)>1:
                ind = []
                for k in range(len(rowedges)):
                    g = grad[rowedges[k],i,:]
                    if g[0] != 0 and g[1]/g[0] <= 1 :
                        ind.append(k)
                rowedges = rowedges[ind]
                if len(rowedges)>1:
                    tie = []
                    for j in range(len(rowedges)):
                        if j!= len(rowedges)-1 and (rowedges[j+1] - rowedges[j]) <= band1:
                            thinedgemat[rowedges[j],i]=0
                            tie.append(rowedges[j])
                        elif len(tie) != 0:
                            thinedgemat[rowedges[j],i]=0
                            tie.append(rowedges[j])
                            #temp = (max(tie) + min(tie))/2
                            temp = sum(tie)/len(tie)
                            thinedgemat[floor(temp),i] = 1
                            thinedgemat[ceil(temp),i] = 1
                            tie = []
            coledges = np.nonzero(edgemat[i,:])[0]
            if len(coledges)>1:
                ind = []
                for k in range(len(coledges)):
                    g = grad[i,coledges[k],:]
                    if g[0] == 0 or g[1]/g[0] > 1 :
                        ind.append(k)
                coledges = coledges[ind]
                if len(coledges)>1:
                    tie = []
                    for j in range(len(coledges)):
                        if j!= len(coledges)-1 and (coledges[j+1] - coledges[j]) <= band1:
                            thinedgemat[i,coledges[j]]=0
                            tie.append(coledges[j])
                        elif len(tie) != 0:
                            thinedgemat[i,coledges[j]]=0
                            tie.append(coledges[j])
                            #temp = (max(tie) + min(tie))/2
                            temp = sum(tie)/len(tie)
                            thinedgemat[i,floor(temp)] = 1
                            thinedgemat[i,ceil(temp)] = 1
                            tie = []
        return thinedgemat


    def __init__(self, image, band_grad, cutoff=0.99, distfunc=Distance):
        super(Edges, self).__init__(image, band_grad, distfunc)
        self.cutoff = cutoff
        self._observed.append('cutoff')
        Edges.reset_Edges(self)
        self.__defined = True

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if self.__defined and name in self._observed:
            if name == "cutoff":
                Edges.reset_Edges(self)
            else:
                Grads.reset_Grads(self)
                Edges.reset_Edges(self)
