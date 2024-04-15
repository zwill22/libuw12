.. libuw12 intro file

.. _`sec:introduction`:

=============
Introduction
=============

Overview
========

LibUW12 is a C++ library for the evaluation of the UW12 Fock matrix and energy, using a combination of density-fitting
and resolution of the identity (RI). This library does not calculate the two- and three-index density-fitting
integrals for either the atomic orbital space or the auxiliary RI space. These therefore must be provided to the
library using the BaseIntegrals class.
Any issues with this library should be raised here; contributions and suggestions are also welcome.

Citing libuw12
================


If you use this library in your program and find it helpful, any feedback would be greatly appreciated.
If you publish results using this library, please cite Z. M. Williams's thesis, which includes details of the
implementation:

[Development of Density Functional Correlation Theories Based on the Unsöld Approximation](https://hdl.handle.net/1983/1584f2c3-21a7-4162-abdc-3aca04e7bfd2),
Williams, Z. M. (Author). 23 Jan 2024

In addition, previous work on UW12 can also be cited:

Z. M. Williams and F. R. Manby, 2021. DOI: [10.26434/chemrxiv-2021-tnw0w](https://doi.org/10.26434/chemrxiv-2021-tnw0w)

Z. M. Williams, T. C. Wiles and F. R. Manby, J. Chem. Theory Comput., 2020, 16, 6176– 6194. DOI: [10.1021/acs.jctc.0c00442](https://doi.org/10.1021/acs.jctc.0c00442)

[Novel, Low-Cost Computational Methods for Predicting the Electronic Structure of Molecules](https://hdl.handle.net/1983/79d98b4e-05f2-44a7-b6e8-d29bec4a8c73),
Wiles, T. C. W. (Author). 28 Nov 2019

T. C. Wiles and F. R. Manby, J. Chem. Theory Comput., 2018, 14, 4590-4599. DOI: [10.1021/acs.jctc.8b00337](https://doi.org/10.1021/acs.jctc.8b00337)

A full bibtex citation can be found in `citation.bib` in the main directory.

Requirements
============

For the library
^^^^^^^^^^^^^^^

- A modern C++ compiler, at least C++17 standard library is required.
- CMake/CTest build tools v. >= 3.22
- Parallelization:
    * Threaded Building Blocks (TBB) C++ template library; (Recommended)
    * OpenMP
- Linear Algebra Library:
    * Armadillo (v9.9 and above) - requires BLAS/MKL backend (Recommended)
    * Eigen (v3.3 and above)
- Catch2 (v3) for testing

For the docs
^^^^^^^^^^^^

- Doxygen
- Sphinx
- Breathe
- Exhale

License
=======

libuw12 is available under an MIT License, allowing for free and open use, reproduction, and modification of the library,
so long as the copyright and license notices are preserved. The authors hold no liability for,
and give no warranty against, results of the use of this software.

Support
=======

If you have any problems or would like to make suggestions for improvements, please raise an issue on the github repo.

Help is always welcome, and if you wish to make contributions to the code yourself, please take a look at the library
API docs and have a go.

.. toctree::
   :hidden:
