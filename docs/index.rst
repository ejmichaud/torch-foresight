.. torch-foresight documentation master file, created by
   sphinx-quickstart on Tue Aug 27 12:28:06 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to torch-foresight's documentation!
===========================================

This package provides a collection of modules useful for characterizing and predicting the dynamics and performance of neural nets. These consist mostly of novel metrics, derived from fields like theoretical neuroscience and information theory, aimed at helping researchers to better understand how neural networks work. The repository is meant to advance a new "Science of AI" or "Science of Deep Learning" (see `neuralnet.science <https://neuralnet.science>`_). It currently includes modules for computing:

* Effective Information (`paper <https://arxiv.org/abs/1907.03902>`_)

With the following under development:

* Gradient Noise Scale (`paper <https://arxiv.org/abs/1812.06162>`_)
* Information Bottleneck (`paper <https://arxiv.org/abs/1503.02406>`_)

.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules/ei
   modules/gns


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
