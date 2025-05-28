from denoise.edge import Edges, Grads
from denoise.lm import Kernel, LocalKfit, Distance, Epanechnikov, LocalSfit
from denoise.utils import Bar
import scipy.cluster.vq as clvq
import numpy as np
import warnings
from joblib import Parallel, delayed, cpu_count
#from numba import jit, cuda
import math, time, multiprocessing
from tqdm_joblib import tqdm_joblib

class ImageEL(Edges):
    __defined=False

    def __init__(self, image, band_edge, band_estim,
                 cutoff=0.99, distfunc_edge=Epanechnikov, distfunc_estim=Epanechnikov):
        super(ImageEL, self).__init__(image, band_edge, cutoff, distfunc_edge)
        self.__defined=False
        self.distfunc_estim = distfunc_estim
        self.band_estim = band_estim
        self._observed_own = ["band_estim", "distfunc_estim"]
        self._observed += self._observed_own
        self.Kernel_estim = Kernel(
            self.rows, self.cols, self.band_estim, self.distfunc_estim)
        self.img_max=np.max(self.image)
        if self.img_max > 100:
            self.img_max = 255
        else:
            self.img_max =1
        self.estimates = ImageEL.LocalDist(self)
        self.__defined = True

    def pad(x, max):
        x = abs(x)
        x = min(x, max - abs(max-x))
        return(int(x))
    def LocalDist(self):
        rows = self.rows
        cols = self.cols
        res = np.empty([rows, cols])
        grad = self.gradmat
        mat = self.edge
        image = self.image
        K = self.Kernel_estim
        b= self.band_estim
        mindist=4
        m1=max(self.band_estim)
        K_clust = Kernel(rows, cols, [mindist,mindist], self.distfunc_estim)
        kernels = Kernel(rows,cols,[2,2],Distance)
        var = self.var
        t1=time.time()
        m=self.img_max
        def func(i,j):
            locs = K.eval([i, j])
            minordist = m1
            majordist = m1
            coord = np.array([i+1, j+1])
            for (a, b) in zip(locs[0], locs[1]):
                a1 = ImageEL.pad(a, rows-1)
                b1 = ImageEL.pad(b, cols-1)
                if mat[a1, b1]:
                    dist = ((a-i)**2 + (b-j)**2)**0.5
                    if (dist < minordist) :
                        minordist = dist
                        coord = np.array([a, b])
            for (a, b) in zip(locs[0], locs[1]):
                a1 = ImageEL.pad(a, rows-1)
                b1 = ImageEL.pad(b, cols-1)
                if mat[a1, b1]:
                    angle1 = np.dot(
                        (coord - [i, j]), (np.array([a, b]) - coord))
                    angle2 = np.dot(
                        (np.array([i, j]) - coord), (np.array([a, b]) + coord - 2*np.array([i, j]))) #migthbe something wrong here -checked
                    if angle1*angle2 > 0:
                        dist2=((a-i)**2 + (b-j)**2)**0.5
                        if (dist2<majordist) and (a!=coord[0]) and (b!= coord[1]) :
                            majordist = dist2
            #print(f"{majordist} -- {minordist}")
            if minordist < mindist :
                locs = np.array(K_clust.eval([i, j]),dtype="int")
                #print(locs[0])
                clwt = 0
                data = []
                for pos in zip(locs[0], locs[1]):
                    data.append([image[ImageEL.pad(pos[0], rows - 1),
                                            ImageEL.pad(pos[1], cols-1)]])
                data = np.array(data, dtype='float64')
                cluster = clvq.kmeans2(data, 4, minit='++')
                def f(pos, currpos):
                    if ((pos[0] == currpos[0] and currpos[1] == pos[1])):
                        return True
                    flag1 = -1
                    flag2 = -2
                    it = 0
                    for a, b in zip(locs[0], locs[1]):
                        if (ImageEL.pad(pos[0], rows-1) == a and
                                ImageEL.pad(pos[1], cols-1) == b):
                            flag1 = cluster[1][it]
                        if (ImageEL.pad(currpos[0], rows - 1) == a and
                                ImageEL.pad(currpos[1], cols - 1) == b):
                            flag2 = cluster[1][it]
                        it = it + 1
                    return flag1 == flag2
                k = K_clust.eval([i, j], f)
                #return max(min(LocalKfit(image, [i, j], k, 1)[0],m),0)
#                return LocalKfit(image, [i, j], k, 1)[0]
                return max(min(LocalSfit(image, [i, j], k, kernels,var,0),m),0)
            else:
                temp = coord - [i, j]
                temp=temp/(sum(temp**2))**0.5
                temp_p=[-temp[1],temp[0]]
                # cosin = temp[1]/(sum(temp**2))**0.5
                # sin = temp[0]/(sum(temp**2))**0.5

                def dist(pos1, pos2, band):
                    v = pos1 - pos2
                    x = temp[0]
                    y = temp[1]
                    # val = (x*cosin+y*sin)**2/(majordist**2 *
                    #                           band[0]**2)+(y*cosin-x*sin)**2/(minordist**2*band[1]**2) #might be something wrong here
                    #val = (x*sin-y*cosin)**2/(majordist**2 *
                    #                          band[0]**2)+(y*sin+x*cosin)**2/(minordist**2*band[1]**2)
                    val = np.dot(v,temp)**2/minordist**2+np.dot(v,temp_p)**2/majordist**2
                    return (val <= 1)*3*(1-val)/4 #somehow using epanechnikov loses performance
                k = Kernel(rows, cols, [1,1], distfunc=dist)
                k = k.eval([i, j])
                return max(min(LocalKfit(image, [i, j], k, 1)[0],m),0)
#                return LocalKfit(image, [i, j], k, 1)[0]
        n_jobs=cpu_count()-1
        batch_size=int(rows*cols/n_jobs)
        with tqdm_joblib(desc="Estimates calculation",total=rows*cols) as progress_bar:
            res=Parallel(n_jobs=n_jobs,prefer="processes",batch_size=batch_size)(delayed(func)(i,j)for i in range(rows) for j in range(cols))
        res=np.array(res,dtype='float64').reshape(image.shape)
        print(f"estimates took {time.time()-t1} seconds.\n")
#       print(f"{rows}__{cols}")
        return res
    def show(self, tp=["Original", "Edges", "Estimates"],
             savefile="temp.png"):
        import matplotlib.pyplot as plt
        dict = {"Original": "image",
                "Edges": "edge",
                "Estimates": "estimates"}
        if len(tp)==1 :
            z=plt.imshow(self.__dict__[dict[tp[0]]],'Greys_r',interpolation='none')
            plt.show()
            return
        fig, axes = plt.subplots(1, len(tp))
        i = 0
        for name in tp:
            a = self.__dict__[dict[name]]
            if name!="Edges":
                axes[i].imshow(a, 'Greys_r',interpolation = "none")
            else:
                axes[i].imshow(a, 'Greys' ,  interpolation = "none")                
            axes[i].axis("off")
            axes[i].set_title(name)
            i += 1
        plt.savefig(savefile)
        # plt.close()
        # c=plt.imshow(self.estimates, 'Greys_r' , interpolation = "none")
        # plt.savefig("result.png")
        try: 
            __IPYTHON__
        except NameError:
            plt.show()
    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if self.__defined and name in self._observed:
            if name in self._observed_own:
                self.Kernel_estim = Kernel(
                    self.rows, self.cols, self.band_estim, self.distfunc_estim)
                ImageEL.LocalDist(self)
            else:
                self.Kernel_estim = Kernel(
                    self.rows, self.cols, self.band_estim, self.distfunc_estim)
                ImageEL.reset_Grads(self)
                ImageEL.reset_Edges(self)
                ImageEL.LocalDist(self)
