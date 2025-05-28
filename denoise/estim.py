from denoise.edge import Edges, Grads
from denoise.lm import Kernel, LocalKfit, Distance
from denoise.utils import Bar

import numpy as np


class Image(Edges):
    def _estim(self):
        image = self.image
        Kernel = self.Kernel_estim
        flt = self.Kernel_filter

        def res(pos):
            f = flt[pos[0], pos[1]]
            k = Kernel.eval(pos, f)
            return(LocalKfit(image, pos, k, 2)[0])
        return res

    def eval(self):
        rows = self.rows
        cols = self.cols
        func = self.estim
        res = np.zeros((rows, cols))
        bar = Bar(max=rows*cols, status="calculating estimates")
        for i in range(rows):
            for j in range(cols):
                res[i, j] = func([i, j])
                bar.next()
        del bar
        return res

    def reset_Estim(self):
        self.list_edges = Image.ListEdges(self)
        self.Kernel_estim = Kernel(
            self.rows, self.cols, self.band_estim, self.distfunc_estim)
        self.estimates = Image.LocalFilter(self)

    def pad(x, max):
        x = abs(x)
        x = min(x, max - abs(max-x))
        return(x)

    def ListEdges(self):
        rows = self.rows
        cols = self.cols
        res = [list(), list()]
        mat = self.edge
        for i in range(rows):
            for j in range(cols):
                if mat[i, j] != 0:
                    res[0].append(i)
                    res[1].append(j)
        return res

    def IsbelowPerp(a, b, c, d):
        def f(pos):
            val = (c*(pos[1]-b) - d*(pos[0]-a))
            return (val >= 0 or np.allclose(val, 0))
        return f

    def LocalFilter(self):
        rows = self.rows
        cols = self.cols
        res = np.empty([rows, cols])
        grad = self.gradmat
        mat = self.edge
        image = self.image
        K = self.Kernel_estim
        run = Bar(max=rows*cols, status="initializing estimates")
        for i in range(rows):
            for j in range(cols):
                locs = K.eval([i, j])
                locedge = np.array([0, 0, np.array([0, 0])])
                for (a, b) in zip(locs[0], locs[1]):
                    a = Image.pad(a, rows-1)
                    b = Image.pad(b, cols-1)
                    if mat[a, b]:
                        locedge = np.vstack([locedge, [a, b, grad[a, b, :]]])
                if len(locedge.shape) == 1:
                    filter1 = Image.IsbelowPerp(0, 0, 0, 0)

                    def filt(pos, currpos):
                        return filter1(pos) == filter1(currpos)
                else:
                    locedge = np.delete(locedge, (0), axis=0)
                    loc_avggrad = np.average(locedge[:, 2])
                    loc_group1 = np.array([0, 0, np.array([0, 0])])
                    loc_group2 = np.array([0, 0, np.array([0, 0])])
                    for row in locedge:
                        if row[2].dot(loc_avggrad) >= 0:
                            loc_group1 = np.vstack([loc_group1, row])
                        else:
                            loc_group2 = np.vstack([loc_group2, row])
                    avg_1 = np.sum(loc_group1, axis=0) / \
                        (loc_group1.shape[0]-(loc_group1.shape[0] == 1))
                    avg_2 = np.sum(loc_group1, axis=0) / \
                        (loc_group2.shape[0]-(loc_group2.shape[0] == 1))
                    filter1 = Image.IsbelowPerp(
                        avg_1[0], avg_1[1], avg_1[2][0], avg_1[2][1])
                    filter2 = Image.IsbelowPerp(
                        avg_2[0], avg_2[1], avg_2[2][0], avg_2[2][1])

                    def filt(posit, currpos):
                        return (filter1(posit) == filter1(currpos) and
                                filter2(posit) == filter2(currpos))
                k = K.eval([i, j], filt)
                res[i, j] = LocalKfit(image, [i, j], k, 1)[0]
                run.next()
        del run
        return res

    __defined = False

    def __init__(self, image, band_edge, band_estim,
                 cutoff=0.99, distfunc_edge=Distance, distfunc_estim=Distance):
        super(Image, self).__init__(image, band_edge, cutoff, distfunc_edge)
        self._observed_own = ["band_estim", "distfunc_estim"]
        self._observed += self._observed_own
        self.distfunc_estim = distfunc_estim
        self.band_estim = band_estim
        Image.reset_Estim(self)
        self.__defined = True

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if self.__defined and name in self._observed:
            if name in self._observed_own:
                Image.reset_Estim(self)
            elif name == "cutoff":
                Edges.reset_Edges(self)
                Image.reset_Estim(self)
            else:
                Grads.reset_Grads(self)
                Edges.reset_Edges(self)
                Image.reset_Estim(self)

    def show(self, type=["Original", "Edges", "Estimates"],
             savefile="temp.png"):
        import matplotlib.pyplot as plt
        dict = {"Original": "image",
                "Edges": "edge",
                "Estimates": "estimates"}
        fig, axes = plt.subplots(1, len(type))
        i = 0
        for name in type:
            a = self.__dict__[dict[name]]
            axes[i].imshow(a, 'Greys_r')
            axes[i].axis("off")
            axes[i].set_title(name)
            i += 1
        plt.savefig(savefile)
        try:
            __IPYTHON__
        except NameError:
            plt.show()
