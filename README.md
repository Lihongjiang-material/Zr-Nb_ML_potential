# Zr-Nb_ML_potential
develop ML potential for Zr-Nb alloys

The ML potential based on the C++ code

=== Installation of the ML package ===

1. Copy all the files in src to LAMMPS/src directory
2. Compile LAMMPS
3. The ML file named Param_ML_pot.txt, it contains all the ML parameters.

=== Run the simple LIT test ===

In the file LIT_example/omega_300k_0GPa_npt, all the necessary files are provided, including LAMMPS in file, omega data file and Param_ML_pot.txt
run the example will see the LIT in the omega.
./lmp_serial < LIT_example/omega_300k_0GPa_npt/in.npt

=== Credits and license ===

This compute was written by H. Li (lihjmaterial@163.com) and H. Zong (zonghust@mail.xjtu.edu.cn) and is licensed under the GPLv2 license.
Please contribute changes back to the community.
