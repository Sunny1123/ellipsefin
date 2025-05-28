Image denoising while preserving edge structure.



To run this implemntation from command line use:

"python main.py &lt;filename&gt; &lt;bw-x-edge&gt; &lt;bw-y-edge&gt; &lt;bw-x-smooth&gt; &lt;bw-y-smooth&gt; &lt;relative weight&gt; &lt;noise level to add&gt; &lt;noise type&gt;"

Path to file should be given relative to the directory containing main.py.

Example Command: python3 main.py ./bin/data/sqcirc.png 2 2 10 10 0.9 5 1

(You may need to use python3 instead of python when running in linux)

  

The results are named according to the following pattern:

"elres &lt;bw-x bw-y&gt; &lt;bw-x-edge&gt; &lt;bw-y-edge&gt; &lt;bw-x-smooth&gt; &lt;bw-y-smooth&gt; &lt;relative weight&gt; &lt;noise level to add&gt; &lt;noise type&gt; &lt;filename&gt;"

The example above generates two files: "elres 2 2 10 10 0.9 5 sqcirc.png", which contains image comparisons, and "elres 2 2 10 10 0.9 5 sqcirc.txt", which provides a summary of the RMSE (Root Mean Square Error) values.

These two files can be found in "./bin/data/runs" folder.

