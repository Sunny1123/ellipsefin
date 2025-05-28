from denoise.edge import Edges, Grads
from denoise.lm import Kernel, LocalKfit, Distance
from denoise.utils import Bar, pad
from scipy.linalg import eigh
import scipy.cluster.vq as clvq
import numpy as np
import math


class ImagePCA(Edges):

	def __init__(self, image, band_edge, band_estim,
				 cutoff=0.99, distfunc_edge=Distance, distfunc_estim=Distance):
		super(ImagePCA, self).__init__(image, band_edge, cutoff, distfunc_edge)
		self.distfunc_estim = distfunc_estim
		self.band_estim = band_estim
		self._observed_own = ["band_estim", "distfunc_estim"]
		self._observed += self._observed_own
#        self.list_edges = ImageEL.ListEdges(self)
		self.Kernel_estim = Kernel(
			self.rows, self.cols, self.band_estim, self.distfunc_estim)
		self.estimates = ImagePCA.LocalDist(self)

	def LocalDist(self):
		rows = self.rows
		cols = self.cols
		res = np.empty([rows, cols])
		grad = self.gradmat
		#print(f"{grad[0,0,:]}")
		mat = self.edge
		print(f"{np.sum(mat)}")
		image = self.image
		K = self.Kernel_estim
		run = Bar(max=rows*cols, status="initializing estimates")
		for i in range(rows):
			for j in range(cols):
				locs = np.array(K.eval([i, j]))
				locs1= []
				for pos in zip(locs[0],locs[1]):
					a = pad(pos[0],rows - 1)
					b = pad(pos[1],cols - 1)
					if mat[a,b]:
						locs1.append([a,b])
				if len(locs1)<=1:
					k = K.eval([i, j])
					res[i, j] = LocalKfit(image, [i, j], k, 2)[0]
					run.next()
				else :
					locs1=np.array(locs1)
					data = []
					for pos in locs1:
						data.append(grad[pos[0],pos[1],:])

					data = np.array(data, dtype='float')
					cluster = clvq.kmeans2(data, 2, minit="points")
					labels =cluster[1]
					labels.shape =(len(labels),1)
					data1 = np.append(locs1, labels,axis=1)
					data = np.append(data1, data,axis=1)
					count1 = sum(data[:,2])
					count2=data.shape[0]-count1
					if count1==0:
						print(" count1 is 0")
						res[i,j] = 1
						#insert 1st cluster empty code
					elif count2==0:
						print("count2 is 0")
						res[i,j] = 1
						#insert 2nd cluster empty code
					else:
						total= np.sum(data[:,0:2],axis=0)
						sum1 = np.sum(np.multiply(data[:,0],data[:,2]))
						sum2 = np.sum(np.multiply(data[:,1],data[:,2]))
						centroid1= np.array([sum1,sum2])
						centroid2 = total-centroid1
						centroid1 = centroid1/count1
						centroid2 = centroid2/count2
						
						data1 = np.array([0,0])
						data1.shape=(1,2)
						data2=data1
						for row in data:
							if row[2]==1:
								data1= np.vstack((data1,np.array([row[3],row[4]])))
							else:
								data2= np.vstack((data2,np.array([row[3],row[4]])))
						data1 = np.delete(data1,0,axis=0)
						data2 = np.delete(data2,0,axis=0)
						if count1==1:
							dir1= np.array([data1[0,1],-data1[0,0]])
							dir1.shape = (2,1)
							s1 = 1
						else:
							val = np.matmul(np.transpose(data1),data1)
							reseig = eigh(val,eigvals=(1,1))
							dir1 = np.array([reseig[1][1],-reseig[1][0]])
							dir1.shape = (2,1)
							s1= reseig[0]
						if count2==1:
							dir2=np.array([data2[0,1],-data2[0,0]])
							dir2.shape = (2,1)
							s2 = 1
						else:
							val = np.matmul(np.transpose(data2),data2)
							reseig = eigh(val,eigvals=(1,1))
							dir2 = np.array([reseig[1][1],-reseig[1][0]])
							dir2.shape = (2,1)
							s2 = reseig[0]
						mat1 = 25*(s1/s2)**2*np.matmul(dir1,np.transpose(dir1)) + 25 * (s2/s1)**2*np.matmul(dir2,np.transpose(dir2))
						print(f"{mat1.shape}-----{dir1}----{dir2}-----{s1}----{s2} \n")
						try:
							mat1 = np.linalg.inv(mat1)
						except np.linalg.LinAlgError:
							print("error")
							#mat1 = np.linalg.pinv(mat1)
						#print(f"{mat1}")
						def dist(pos1 , pos2, band=[1,1]):
							val = pos1 - pos2
							val.shape = (2,1)
							d = np.matmul(np.transpose(val),mat1)
							d = np.matmul(d,val)
							d.shape = ()
							d= math.exp(-d/2 )
							return d
						k = Kernel(rows, cols, self.band_estim, distfunc = dist).eval(np.array([i,j]))
						print(f"{k}")
						res[i,j] = LocalKfit(image, [i, j], k, 1)[0]
						#print(f"c1{data1}--\nc2{data2}---\ns2{sum2}----s1{sum1}")
						run.next()

					
		del run
		return res
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
		    if name!="Edges":
		        axes[i].imshow(a, 'Greys_r')
		    else:
		        axes[i].imshow(a, 'Greys')                
		    axes[i].axis("off")
		    axes[i].set_title(name)
		    i += 1
		plt.savefig(savefile)
		try:
		    __IPYTHON__
		except NameError:
		    plt.show()