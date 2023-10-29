.. DHL-LinAlg documentation master file, created by
   sphinx-quickstart on Tue Aug 29 06:39:02 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DHL-LinAlg documentation!
====================================

DHL-LinAlg is a simple linear algebra implementation in C++ providing classes such as **Vector** and **Matrix**.
All classes are binded into python.


Quickstart
===============

Installation can be done as follows:

..  code-block:: bash

   pip install dhllinalg 


Afterwards you can import the library in python

..  code-block:: python

   from dhllinalg.bla import Matrix

   m = Matrix(10,10)
   for i in range(10):
      for j in range(10):
         m[i,j] = i + 2 * j

   print(m)




========
Contents
========

.. toctree::
   ← Back to Github <https://github.com/DHL-ASC/DHL-LinAlg>
   
.. toctree::
   :maxdepth: 1
   :caption: Contents:

   install
   examples/index
   api_reference

.. toctree::
   :maxdepth: 1
   :caption: Information:

   changelog/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
