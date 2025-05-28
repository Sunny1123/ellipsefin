Image denoising while preserving edge structure.


To run this implemntation from command line use:

"python main.py <filename> <bw-x-edge> <bw-y-edge> <bw-x-smooth> <bw-y-smooth> <relative weight> <noise level to add> <noise type>"

Path to file should be given relative to the directory containing main.py.

from command prompt/bash.(You may need to use python3 instead of python when running in linux)

Example Command: python3 main.py ./bin/data/test.png 2 2 10 10 0.9 5 1
  
The results are named according to the following pattern:

"elres <bw-x bw-y> <bw-x-edge> <bw-y-edge> <bw-x-smooth> <bw-y-smooth> <relative weight> <noise level to add> <noise type> <filename>"
  e.g: The example above generates result "elres 2 2 10 10 0.9 5 clres.png" and "elres 2 2 10 10 0.9 5 clres.txt"

These two files can be found in "./bin/data/runs" folder.

