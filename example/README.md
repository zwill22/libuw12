# UW12 Example

This directory contains a working example for a water molecule. Two- and
three-index density fitting integrals are all pre-calculated and taken
from the `test_data` directory.

## Build

To build this, make sure you have built libuw12. Then using cmake in this directory

```
cmake . -Bbuild -DCMAKE_LIBRARY_PATH="/path/to/directory/containing/libuw12.so" -DCMAKE_INCLUDE_PATH="/path/to/directory/containing/libuw12.hpp"
```
where the -I and -L flags are used to point to the libuw12 header and library, respectively, if these are not already in your path.

To then build the library and run the test:
```
cd build
make
```
