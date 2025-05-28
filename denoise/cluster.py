import numpy as np
import sklearn.cluster as cl
import scipy.cluster.vq as clvq
from denoise.lm import Kernel, LocalKfit
from denoise.utils import Bar


class ImageCL(object):
    def __init__(self, image, band, clwt=0, cutoff=0.01):
        self.image = image
        self.band = band
        self.clwt = clwt/(band[0]**2+band[1]**2)**(1/2)
        self.Kernel = Kernel(self.image.shape[0], self.image.shape[1], band)
        self.cutoff = ImageCL.Cutoff_stdev(self, cutoff)
        self.estimates = ImageCL.evalfilter(self)

    def pad(x, max):
        x = abs(x)
        x = min(x, max - abs(max-x))
        return(x)

    def Cutoff_stdev(self, cutoff):
        image = self.image
        K = self.Kernel
        mat = np.zeros(image.shape)
        rows, cols = image.shape
        for i in range(rows):
            for j in range(cols):
                ker = K.eval([i, j])
                ind0 = np.array(ker[0])
                ind1 = np.array(ker[1])
                for k in range(len(ind0)):
                    ind0[k] = ImageCL.pad(ind0[k], rows-1)
                    ind1[k] = ImageCL.pad(ind1[k], cols-1)
                try:
                    data = image[ind0, ind1]
                except IndexError:
                    print(f"{ind0}----{ind1}----{ker}")
                mat[i, j] = np.std(data)
        res = np.quantile(mat, cutoff)
        return(res)

    def evalfilter(self):
        rows, cols = self.image.shape
        res = np.empty([rows, cols])
        K = self.Kernel
        run = Bar(max=rows*cols, status="initializing estimates")
        image = self.image
        clwt = self.clwt
        for i in range(rows):
            for j in range(cols):
                locs = np.array(K.eval([i, j]))
                data = []
                for pos in zip(locs[0], locs[1]):
                    data.append([pos[0]*clwt, pos[1]*clwt,
                                 self.image[ImageCL.pad(pos[0], rows - 1),
                                            ImageCL.pad(pos[1], cols-1)]])
                data = np.array(data, dtype='float')
                #cluster = clvq.kmeans2(data, 2, minit='points')
                cluster = cl.DBSCAN(eps=0.1).fit(data)
                stdevflag = np.std(data[:, 2]) < self.cutoff

                def f(pos, currpos):
                    if ((pos[0] == currpos[0] and currpos[1] == pos[1])
                            or stdevflag):
                        return True
                    flag1 = -1
                    flag2 = -2
                    it = 0
                    for a, b in zip(locs[0], locs[1]):
                        if (ImageCL.pad(pos[0], rows-1) == a and
                                ImageCL.pad(pos[1], cols-1) == b):
                            #flag1 = cluster[1][it]
                            flag1 = cluster.labels_[it]
                        if (ImageCL.pad(currpos[0], rows - 1) == a and
                                ImageCL.pad(currpos[1], cols - 1) == b):
                            #flag2 = cluster[1][it]
                            flag2 = cluster.labels_[it]
                        it = it + 1
                    return flag1 == flag2
                k = K.eval([i, j], f)
                res[i, j] = LocalKfit(image, [i, j], k, 1)[0]
                run.next()
        del run
        return res

    def _estim(self):
        image = self.image
        Kernel = self.Kernel
        flt = self.filter

        def res(pos):
            f = flt[pos[0], pos[1]]
            print(f"{f}__{pos}\n")
            k = Kernel.eval(pos, f)
            return(LocalKfit(image, pos, k, 2)[0])
        return res

    def eval(self):
        rows, cols = self.image.shape
        res = np.zeros(rows*cols).reshape(rows, cols)
        estim = ImageCL._estim(self)
        for i in range(rows):
            for j in range(cols):
                res[i, j] = estim([i, j])
        return res

    def show(self, savefile="temp.png"):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(self.estimates, "Greys_r", interpolation="bilinear")
        axes[0].set_title("Estimate")
        axes[0].axis("off")
        axes[1].imshow(self.image, "Greys_r", interpolation="bilinear")
        axes[1].set_title("Original")
        axes[1].axis("off")
        plt.savefig(savefile)
        try:
            __IPYTHON__
        except NameError:
            plt.show()
