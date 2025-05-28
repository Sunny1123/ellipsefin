Image denoising while preserving edge structure.
Currently I implemeted two methods.
First method calculates the edges by estimating gradients using 2nd order taylor expansions and OLS.
Then divides eachneighbourhood into sector using the information about edges and estimates using the points in the sector where the original datapoint resides.
Second method cluster each neghborhood into two cluster using K means on paired coordinates and image intensity. And estimates image intensity at a point using the cluster the point belongs two.
Both method uses a first order taylor expansion around each pixel to calculate the estimates.

You can also run the code from command line using:
"python main.py <filename> <bw-x-edge> <bw-y-edge> <bw-x-smooth> <bw-y-smooth> <relative weight> <noise level to add> <noise type>"

Path to file shoul be given relative to the directory containing main.py.

from command prompt/bash.(You might need to use python3 instead of python when running in linux)
e.g: python main.py ./bin/data/test.png 2 2 10 10 0.9 5 1
  
The results are named according to the following pattern:

"elres <bw-x bw-y> <bw-x-edge> <bw-y-edge> <bw-x-smooth> <bw-y-smooth> <relative weight> <noise level to add> <noise type> <filename>"
  e.g: The example above generates result "elres 2 2 10 10 0.9 5 clres.png"
