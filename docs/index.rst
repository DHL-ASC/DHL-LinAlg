.. DHL-LinAlg documentation master file, created by
   sphinx-quickstart on Tue Aug 29 06:39:02 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DHL-LinAlg documentation!
====================================

DHL-LinAlg is a simple linear algebra implementation using C++ providing template classes such as **Vector** and **Matrix**.

Installation and running test examples can be done as follows:

..  code-block::
    
    git clone https://github.com/DHL-ASC/DHL-LinAlg.git
    cd DHL-LinAlg
    pip install . 


To implement your own C++ tests with these classes you can edit and run the files from the "DHL-LinAlg/tests" folder:

..  code-block::

    cd DHL-LinAlg
    mkdir build
    cd build
    cmake ..
    make
    
or include the header files

..  code-block::

    #include <vector.h>
    #include <matrix.h>

All objects are implemented in the namespace bla. To use them with less typing, you can set

..  code-block::
    
    using namespace bla;

Vector template
===============

You can create vectors and compute with vectors like:

..  code-block:: cpp
                 
   Vector<double> x(5), y(5), z(5);
   for (int i = 0; i < x.Size(); i++)
      x(i) = i;
   y = 5.0
   z = x+3*y;
   cout << "z = " << z << endl;

Matrix template
===============

For matrices you can choose between row-major (`RowMajor`) or column-major (`ColMajor`) storage,
default is row-major.

..  code-block:: cpp

   Matrix<double,RowMajor> m1(5,3), m2(3,3);
   for (int i = 0; i < m1.Height(); i++)
     for (int j = 0; j < m1.Width(); j++)
       m1(i,j) = i+j;
   m2 = 3.7;
   Matrix product = m1 * m2;
   
You can extract a rows or a columns from a matrix:

..  code-block:: cpp

   Vector col1 = product.Col(1);


some changes ...  

========
Contents
========

.. toctree::
   ← Back to Github <https://github.com/DHL-ASC/DHL-LinAlg>
   
.. toctree::
   :maxdepth: 1
   :caption: Contents:

   install
   examples

.. toctree::
   :maxdepth: 1
   :caption: Information:

   changelog/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
