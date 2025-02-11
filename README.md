# IntroPARCO_H2
<p>To run the simulation on the cluster: </p>
<p>Load the modules:</p>
<p>module load gcc91</p>
<p>module load mpich-3.2.1--gcc-9.1.0</p>
<p>Compile: mpicc transpose_rows.c -o rows -fopenmp</p>
<p>And run with a number of processors=N and size=SIZE: mpirun -np N ./rows SIZE</p>
<p>The number of processors requested for the experiment was 64, the memory was 4 Gb</p>
<p>To run using the .pbs:</p>
<p>change the cd command to the directory where the code is</p>
<p>Run: dos2unix parco_h2.pbs</p>
<p>Submit the job: qsub parco_h2.pbs</p>

<p>The send_recv.c file is the same as mpi_row_block.c but with another version of the transposition routine that I used to compare the approaches</p>
<p>To compile and run it:</p>
<p>mpicc send_recv.c -o sr -fopenmp</p>
<p>mpirun -np N ./sr SIZE</p>
<p>does not work when the number of processors exceed the matrix size</p>
