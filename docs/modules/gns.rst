
Gradient Noise Scale
====================


Theory
^^^^^^

The `Gradient Noise Scale <https://arxiv.org/abs/1812.06162>`_ is a statistical measure which roughly predicts the optimal batch size for a given training task. During stochastic gradient descent, weight updates are performed using a gradient which is not computed from the whole dataset, but from a fraction of it (by randomly sampling a finite number of samples and averaging the gradient that each of these individually produce). The variance in the gradient between batches indicates how complex the dataset is. If this variance is high, then one should try to improve the accuracy of the stochastic gradient by increasing the batch size. A simplified version of this metric can be defined:

.. math::
	\B_\text{simple} = \frac{\text{tr}(\Sigma)}{\norm{G}^2}


Code Documentation
^^^^^^^^^^^^^^^^^^

.. automodule:: foresight.gns
   :members: