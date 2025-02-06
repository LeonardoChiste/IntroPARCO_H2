# IntroPARCO_H2
<p>To run the simulation on the cluster: </p>
<p>Load the modules:</p>
<p>module load gcc91</p>
<p>module load mpich-3.2.1--gcc-9.1.0</p>
<p>Compile: mpicc transpose_rows.c -o rows -fopenmp</p>
<p>And run with a number of threads=N and size=SIZE: mpirun -np N ./rows SIZE</p>
<p>The number of processors requested for the experiment was 64, the memory was 4 Gb</p>
<p>To run using the .pbs:</p>
<p>change the cd command to the directory where the code is</p>
<p>Run: dos2unix parco_h2.pbs</p>
<p>Submit the job: qsub parco_h2.pbs</p>
