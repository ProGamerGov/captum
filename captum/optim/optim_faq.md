---
id: faq
title: FAQ
---


* [How do I know if my model is compatible with the Optim?](#how-do-i-know-if-my-model-is-compatible-with-the-optim)
* [Are only 3 channel RGB images and 4 channel RGBA images supported and can I use a different color space?](#are-only-3-channel-rgb-images-and-4-channel-rgba-images-supported-and-can-i-use-a-different-color-space)
* [Why are my rendered visualizations poor quality or non-existent in outputs?](#why-are-my-rendered-visualizations-poor-quality-or-non-existent-in-outputs)
* [Does the Optim module support JIT?](#does-the-optim-module-support-jit)
* [What dtypes does the Optim module support?](#what-dtypes-does-the-optim-module-support)
* [How can I avoid out of memory errors when rendering?](#how-can-i-avoid-out-of-memory-errors-when-rendering)
* [Can I use Optim with a CPU or do I have to use a GPU?](#can-i-use-optim-with-a-cpu-or-do-i-have-to-use-a-gpu)
* [Do I have to use the provided transforms or can I use transforms from other libraries?](#do-i-have-to-use-the-provided-transforms-or-can-i-use-transforms-from-other-libraries)
* [Does Optim work with torch.fx?](#does-optim-work-with-torchfx)
* [Does Optim support TensorBoard?](#does-optim-support-tensorboard)


## Optim


### **How do I know if my model is compatible with the Optim?**

In general model layers need to be [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)s as functional layers don't support hooks. Please check out the 'Getting Started Model Preparation' tutorial notebook for more information.

### **Are only 3 channel RGB images and 4 channel RGBA images supported and can I use a different color space?**

By default the rendering modules in Optim are setup for rendering [RGB](https://en.wikipedia.org/wiki/RGB_color_spaces) / [RGBA](https://en.wikipedia.org/wiki/RGBA_color_model) images, but they can easily support other [color spaces](https://en.wikipedia.org/wiki/Color_space) with a simple settings change. In the case of ``ToRGB``, you may have to provide a new 3x3 transform matrix for 3 channel (with an optional 4th alpha channel) color spaces. For color spaces using less than or greater than 3 channels, you will need to create a custom color recorrelation module to replace ``ToRGB``. New color correlation matrices can be created using the dataset module, or with your own custom algorithms.

### **Why are my rendered visualizations poor quality or non-existent in outputs?**

There are a wide array of factors that dictate how well a model performs for rendering visualizations. Aspects like the model architecture, the training data used to train the model, the optimizer being used, and your Optim module settings like parameterizations & transforms all play an important role in creating visualizations.

ReLU layers will block the flow of gradients during the backward pass, if their inputs are less than 0. This can result in no visualizations being produced for the target, even if the model already performs well with other targets. To avoid this issue, you can ensure that all applicable ReLU layers have been replaced with Optim's ``RedirectedReLU`` layer (the ``replace_layers`` function makes this extremely easy to do!).

### **Does the Optim module support JIT?**

For the most part, yes. Image parameterizations, transforms, and many of the helper classes & functions support [JIT / TorchScript](https://pytorch.org/docs/stable/jit.html). The provided models also support JIT, but rendering JIT models with ``InputOptimizatization`` is not supported. The ``InputOptimizatization`` class itself does not support JIT either, but it does work with scripted image parameterizations and transforms. The loss objective system also does not support JIT. These limitations are due to the limitations with JIT supporting PyTorch hooks.

### **What dtypes does the Optim module support?**

By default, the ``torch.float32`` / ``torch.float`` dtype is used for all rendering modules. Varying levels of support exist for the other non default [float dtypes](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) (``torch.float64`` / ``torch.double``, ``torch.float16`` / ``torch.half``, & ``torch.bfloat16``). For example, the Optim module easily supports ``torch.float64``, while FP16 support is sporadic. 

There are currently multiple limitations for ``torch.float16`` & ``torch.bfloat16`` support, due to PyTorch's ongoing work on implementing support for these dtypes. These limitations include:

* The ``FFTImage`` parameterization currently only works with ``torch.float16`` & CUDA on PyTorch v1.12.0 and above using size values that are powers of 2.
* The ``RandomScale`` and ``RandomSpatialJitter`` transforms do not currently support ``torch.float16``.
* The ``RandomRotation`` and ``RandomScaleAffine`` transforms currently only support ``torch.float16`` with CUDA.
* The ``ToRGB`` transform's inverse parameter does not currently support ``torch.float16`` or ``torch.bfloat16``.

These limitations can be partially overcome by utilizing PyTorch's [Automatic Mixed Precision package](https://pytorch.org/docs/stable/amp.html) (AMP)'s ``torch.autocast`` context manager. However, rendering may not work correctly unless you are using ``torch.float32`` / ``torch.float`` or ``torch.float64`` / ``torch.double``.

### **How can I avoid out of memory errors when rendering?**

If you are getting out of memory (OOM) errors when trying to render visualizations, you may have to reduce the batch size and / or size of the image parameterization being used. If you are using a custom module, then you should make sure that there are no memory leaks present in it.

### **Can I use Optim with a CPU or do I have to use a GPU?**

Yes, Optim will work with a CPU though it will be slower than when using a GPU.

### **Do I have to use the provided transforms or can I use transforms from other libraries?**

No, you can use transforms from other libraries like [Torchvision](https://pytorch.org/vision/stable/index.html) and [Kornia](https://kornia.github.io/). You can also create your own custom transforms! The only restriction on transforms is that they must be autograd compatible.

### **Does Optim work with torch.fx?**

No, the Optim module's dynamic control flow is not supported by [torch.fx](https://pytorch.org/docs/stable/fx.html). Custom modules though may be able to have some level of support for ``torch.fx``.

This means that Torchvision's ``torchvision.models.feature_extraction`` [package](https://pytorch.org/vision/stable/feature_extraction.html) won't work as it requires ``torch.fx`` support. The Optim module's ``collect_activations`` function should be used instead.

### **Does Optim support TensorBoard?**

Yes, Optim supports [TensorBoard](https://www.tensorflow.org/tensorboard) via PyTorch's ``torch.utils.tensorboard`` [module](https://pytorch.org/docs/stable/tensorboard.html). You will have to add PyTorch's TensorBoard related functions to your code first before using TensorBoard.
