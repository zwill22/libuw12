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
