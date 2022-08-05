Transforms
=================================

.. automodule:: captum.optim.transforms

Alpha Channel Specific Transforms
---------------------------------
.. autoclass:: BlendAlpha
    :members:
.. autoclass:: IgnoreAlpha
    :members:

Random Transforms
---------------------------------

.. autoclass:: RandomSpatialJitter
    :members:
.. autoclass:: RandomScale
    :members:
.. autoclass:: RandomScaleAffine
    :members:
.. autoclass:: RandomRotation
    :members:
.. autoclass:: RandomCrop
    :members:
.. autoclass:: TransformationRobustness
    :members:

Image Correlation
---------------------------------
.. autoclass:: ToRGB
    :members:

Model Preprocessing
---------------------------------
.. autoclass:: ScaleInputRange
    :members:
.. autoclass:: RGBToBGR
    :members:

Other Image Transforms
---------------------------------
.. autoclass:: CenterCrop
    :members:
.. autofunction:: center_crop
.. autoclass:: GaussianSmoothing
    :members:
.. autoclass:: SymmetricPadding
    :members:
.. autoclass:: NChannelsToRGB

Text Transforms
---------------------------------
.. autoclass:: CLIPTokenizer
    :members:
