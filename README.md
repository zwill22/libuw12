# LibUW12

[![C++][cpp-badge]][cpp]
[![CMake][cmake-badge]][cmake]
[![GitHub][github-badge]][github]
[![GitHub Actions][github-actions-badge]][github-actions]
[![Tests][test-badge]][tests]
[![Read the Docs][rtd-badge]][rtd]
[![Documentation Status][doc-badge]][docs]
[![License][license-badge]][license]

LibUW12 is a C++ library for the evaluation of the UW12 Fock matrix and energy, using a combination of density-fitting
and resolution of the identity (RI). This library does not calculate the two- and three-index density-fitting
integrals for either the atomic orbital space or the auxiliary RI space. These therefore must be provided to the
library using the `BaseIntegrals` class.
Any issues with this library should be raised here; contributions and suggestions are also welcome.

## Dependencies

- A modern C++ compiler, at least C++17 standard library is required. This has been tested with:
  - gcc (v11.4.0 and above)
- CMake/CTest build tools (v3.22 and higher)
- Parallelization:
  - Threaded Building Blocks (TBB) C++ template library; or
  - OpenMP
- Linear Algebra Library:
  - Armadillo (v9.9 and above) - requires BLAS/MKL backend
  - Eigen (v3.3 and above)
- Catch2 (v3) for testing

## Documentation

Please refer to the main documentation [here][docs].

## Examples

Examples are included in the example folder.

## Acknowledging usage

If you use this library in your program and find it helpful, any feedback would be greatly appreciated.
If you publish results using this library, please cite Z. M. Williams's thesis, which includes details of the
implementation:

[Development of Density Functional Correlation Theories Based on the Unsöld Approximation][thesis],
Williams, Z. M. (Author). 23 Jan 2024

In addition, previous work on UW12 can also be cited:

**Optimization of a Range-Separated UW12 Hybrid Functional**, Z. M. Williams and F. R. Manby, 2021. DOI: [10.26434/chemrxiv-2021-tnw0w][preprint]

**Accurate Hybrid Density Functionals with UW12 Correlation**, Z. M. Williams, T. C. Wiles and F. R. Manby, J. Chem. Theory Comput., 2020, 16, 6176– 6194. DOI: [10.1021/acs.jctc.0c00442][paper]

[Novel, Low-Cost Computational Methods for Predicting the Electronic Structure of Molecules][thesis-tim],
Wiles, T. C. W. (Author). 28 Nov 2019

**Wavefunction-like Correlation Model for Use in Hybrid Density Functionals**, T. C. Wiles and F. R. Manby, J. Chem. Theory Comput., 2018, 14, 4590-4599. DOI: [10.1021/acs.jctc.8b00337][paper-tim]

A full bibtex citation can be found in `citation.bib` in the main directory.

<!-- Badges -->

[cpp-badge]: https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white
[cmake-badge]: https://img.shields.io/badge/CMake-%23008FBA.svg?style=for-the-badge&logo=cmake&logoColor=white
[github-badge]: https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white
[github-actions-badge]: https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white
[test-badge]: https://img.shields.io/github/actions/workflow/status/zwill22/libuw12/test.yml?style=for-the-badge&logo=github
[rtd-badge]: https://img.shields.io/badge/Read%20the%20Docs-8CA1AF?logo=readthedocs&logoColor=fff&style=for-the-badge
[doc-badge]: https://img.shields.io/readthedocs/libuw12?style=for-the-badge&logo=readthedocs
[license-badge]: https://img.shields.io/github/license/zwill22/libuw12?style=for-the-badge

<!-- Links -->

[cpp]: https://cppreference.com/
[github]: https://github.com/zwill22/libuw12
[github-actions]: https://github.com/zwill22/libuw12/actions
[tests]: https://github.com/zwill22/libuw12/actions/workflows/tests.yml
[rtd]: https://about.readthedocs.com/
[docs]: https://libuw12.readthedocs.io/en/latest/?badge=latest
[license]: https://github.com/zwill22/libuw12/blob/main/LICENSE
[cmake]: https://cmake.org/
[thesis]: https://research-information.bris.ac.uk/en/studentTheses/development-of-density-functional-correlation-theories-based-on-t/
[preprint]: https://doi.org/10.26434/chemrxiv-2021-tnw0w
[paper]: https://doi.org/10.1021/acs.jctc.0c00442
[thesis-tim]: https://research-information.bris.ac.uk/en/studentTheses/novel-low-cost-computational-methods-for-predicting-the-electroni/
[paper-tim]: https://doi.org/10.1021/acs.jctc.8b00337
