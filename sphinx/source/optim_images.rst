Images
====================================

.. automodule:: captum.optim.images

Tensors
------------------------------------

.. autoclass:: ImageTensor
    :members:

Base Classes
------------------------------------

.. autoclass:: InputParameterization
.. autoclass:: ImageParameterization

Image Parameterizations
------------------------------------

Image parameterizations store parameters that require grad that they then return as full
NCHW image(s) when run.

.. autoclass:: FFTImage
    :members:
.. autoclass:: PixelImage
    :members:
.. autoclass:: LaplacianImage
    :members:

Image Parameterization Augmentations
------------------------------------

Image Parameterizations can be augmented with special ``ImageParameterization`` classes
that are intialized with other ``ImageParameterization`` instances.

.. autoclass:: SharedImage
    :members:
.. autoclass:: StackImage
    :members:

Image Parameterization Optimization
------------------------------------

Special ``ImageParameterization`` classes that make optimizing the parameterization
easier.

.. autoclass:: NaturalImage
    :members:
