.. libuw12 install file

.. _`sec:installation`:

=====================================
Installation
=====================================

Obtaining libuw12
===================

The latest stable release of libue12 can always be found at the Github Repo_.

.. _Repo: https://github.com/zwill22/libuw12

It can be downloaded directly from there, or you can clone it locally using git with the command

.. code-block:: bash

	git clone https://github.com/zwill22/libuw12

If you are a developer looking to make changes to the code, please fork the repo into your own version,
and make a pull request when you think your changes are production ready.

Building
========

To build the library, do the following in the top of the source tree:

.. code-block:: bash

	mkdir build
	cd build
	cmake [options] ..
	make

Documentation
^^^^^^^^^^^^^

This documentation can be generated locally via CMake by running

.. code-block:: bash

	make docs

This requires the following to be available:

- Doxygen
- Sphinx
- Breathe
- Exhale

Testing
=======

To run all the tests, in the build directory run

.. code-block:: bash

	make test



.. toctree::
   :hidden:

