# NPTLS 2025: Ab initio Hartree-Fock calculations of nuclei

This repository contains lecture and exercise materials for the part of the 2025 Nuclear Physics Turtle Lecture Series on "Ab initio Hartree-Fock calculations of nuclei." It is structured as follows:

- `lectures/` contains material related to the lectures, including slides and handwritten notes.
- `exercises/` contains Jupyter notebooks for the exercises for the students.
- `nptlshf/` contains source code related to Hartree-Fock calculations and to provide utilities for the exercises. These utilities are intended to be used, but not modified for the exercises. Of course, curious students may look at the code here and experiment with it if they choose.
- `data/` contains data related to matrix elements necessary to construct the Hamiltonians for nuclei.

## Exercises

The exercises are presented as Jupyter notebooks that students work through interactively to learn the fundamentals and the implications of Hartree-Fock calculations for nuclei. See [Exercises](exercises/) for more details. Reference solutions (as a guide, but not the only valid solution) will be provided after each session.

Students should ensure that the have a working installation of Python3 (version 3.6 or higher) with the packages `numpy`, `scipy`, `matplotlib`, and `jupyter` installed.

Students can run the exercises by executing
```
jupyter notebook
```
from the current directory (*not the `exercises` directory).
This should open their web browser, where students can navigate to the `exercises` directory and see the notebooks for the various exercises.

## Larger runs

For runs with emax larger than 4, one needs to take advantage of the symmetries of the HF calculation, which make the problem very sparse. A helper class `HFHelper` is provided to allow you to compute energy expectation values and Fock matrices for up to `emax=8`. An example of how to use this class is given in [`large_scale_hf_example.py`](large_scale_hf_example.py).

## Data

For larger problems, the data we require is too large to store on Github without compression. All of the matrix elements in `data/emax_8` are stored as zipped files. Before trying to run calculations using `HFHelper`, please unzip all files in `data/emax_8` (this will use a total of 6.1 GB):

```
cd data/emax_8/
for f in *.mmh.zip; do unzip $f; done
cd ../..
```

## Contact

Feel free to contact me at heinz.matthias.physics AT gmail.com with any concerns or questions.
