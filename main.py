from denoise import utils
from denoise import cluster
from denoise import ellipse
import matplotlib.image as mpimg
from sys import argv
from matplotlib.pyplot import imshow, show, title, colorbar, annotate
from matplotlib import pyplot as plt
import numpy as np
import cv2 
import math
import platform

np.seterr(all="raise")
filename = argv[1] 
band1 = [float(argv[2]), float(argv[3])] # band-width for edge detection
band2 = [int(argv[4]), int(argv[5])] # band-width for smoothing
weight = float(argv[6]) # Cutoff for edge detection between 0 and 1
noise = float(argv[7]) #Noise sd between 0 to 255
ntype= int(argv[8]) # 1 : gaussian, 2: uniform, 3: laplace 
n_iter= 1

#read Image
img = cv2.imread(filename,0)
if len(img.shape) > 2:
    img = img[:,:,0]
rows, cols = img.shape

#add noise
img1 = utils.Add_Noise(img, noise, 3)

#Iteration; set to 1
it = []
it.append(ellipse.ImageEL(img1, band1,band2, weight))
for i in range(n_iter-1):
    it.append(ellipse.ImageEL(it[i].estimates, band1,band2, weight))
res = it[n_iter-1]
if platform.system()== "Windows" :
    f=filename.split('\\')
    s=f[3].split('.')
    savefile= f[0]+"\\"+f[1]+"\\"+f[2]+f"\\runs\\elres {band1[0]} {band1[1]} {band2[0]} {band2[1]} {weight} {noise} "+ s[0] + ".png"
    savefile1= f[0]+"\\"+f[1]+"\\"+f[2]+f"\\runs\\elres {band1[0]} {band1[1]} {band2[0]} {band2[1]} {weight} {noise} "+ s[0] + ".txt"
else:
    f=filename.split('/')
    s=f[2].split('.')
    savefile= f[0]+"/"+f[1]+f"/runs/elres {band1[0]} {band1[1]} {band2[0]} {band2[1]} {weight} {noise} "+ s[0] + ".png"
    savefile1= f[0]+"/"+f[1]+f"/runs/elres {band1[0]} {band1[1]} {band2[0]} {band2[1]} {weight} {noise} "+ s[0] + ".txt"


#Plotting
fig, axes = plt.subplots(1,5)
axes[0].imshow(img,"Greys_r",interpolation="None")
axes[0].axis("off")
axes[0].set_title("Orginal")
axes[1].imshow((img - img1)**2,"Greys", interpolation ="None")
axes[1].axis("off")
axes[1].set_title("Noise")
axes[2].imshow(img1,"Greys_r",interpolation="None")
axes[2].axis("off")
axes[2].set_title("Noisy")
axes[3].imshow(res.estimates,"Greys_r",interpolation="None")
axes[3].axis("off")
axes[3].set_title("Denoised")
axes[4].imshow(res.edge,"Greys", interpolation ="None")
axes[4].axis("off")
axes[4].set_title("Edges")

plt.show()
plt.savefig(savefile)
#print(f"{np.max(img)}")
max = 1
if np.max(img)>20:
    max = 255
imshow((img - img1)**2,"Greys")
title("Original - noisy")
colorbar()
show()
print(f" root mean sum square of difference between original and noise image : \n {(np.sum((img - img1)**2)/(rows*cols))**(0.5)}")
print(f"\n PSNRnoise = {10*math.log((max**2/(np.sum((img - img1)**2)/(rows*cols))),10)}")
imshow((img - res.estimates)**2,"Greys")
title("original - estimates")
colorbar()
show()
print(f"root mean sum square of difference between original and estimate image : \n {(np.sum((img - res.estimates)**2)/(rows*cols))**(0.5)}")
print(f"\n PSNResti = {10*math.log((max**2/(np.sum((img - res.estimates)**2)/(rows*cols))),10)}")
imshow(np.multiply((img - img1)**2,(1-res.edge)),"Greys")
title("Original - noisy, no edge")
colorbar()
show()
print(f"root mean sum square of difference between original and noise image (no edge) : \n {(np.sum(np.multiply((img - img1)**2,(1-res.edge)))/np.sum(1-res.edge))**(0.5)}")

imshow(np.multiply((img - res.estimates)**2,(1-res.edge)),"Greys")
title("original - estimates, no edge")
colorbar()
show()
print(f"root mean sum square of difference between original and estimate image (no edge) : \n {(np.sum(np.multiply((img - res.estimates)**2,(1-res.edge)))/np.sum(1-res.edge))**(0.5)}")

print("Summary betwee original and estimates\n")
print(f"{utils.summary(img,res.estimates)}")
print("Summary between original and noise\n")
print(f"{utils.summary(img,img1)}")
f= open(savefile1,"w")
f.write("Summary between original and estimates\n" + utils.summary(img,res.estimates) + "\nSummary between original and noise\n" +
utils.summary(img,img1) ) 
f.close()