---
id: faq
title: FAQ
---




* **[Attribution](#attribution)**

  * [How do I set the target parameter to an attribution method?](#how-do-i-set-the-target-parameter-to-an-attribution-method)
  * [I am facing Out-Of-Memory (OOM) errors when using Captum. How do I resolve this?](#i-am-facing-out-of-memory-oom-errors-when-using-captum-how-do-i-resolve-this)
  * [I am using a perturbation based method, and attributions are taking too long to compute. How can I speed it up?](#i-am-using-a-perturbation-based-method-and-attributions-are-taking-too-long-to-compute-how-can-i-speed-it-up)
  * [Are SmoothGrad or VarGrad supported in Captum?](#are-smoothgrad-or-vargrad-supported-in-captum)
  * [How do I use Captum with BERT models?](#how-do-i-use-captum-with-bert-models)
  * [My model inputs or outputs token indices, and when using Captum I see errors relating to gradients, how do I resolve this?](#my-model-inputs-or-outputs-token-indices-and-when-using-captum-i-see-errors-relating-to-gradients-how-do-i-resolve-this)
  * [Can my model using functional non-linearities (E.g. nn.functional.ReLU) or reused modules be used with Captum?](#can-my-model-using-functional-non-linearities-eg-nnfunctionalrelu-or-reused-modules-be-used-with-captum)
  * [Do JIT models, DataParallel models, or DistributedDataParallel models work with Captum?](#do-jit-models-dataparallel-models-or-distributeddataparallel-models-work-with-captum)
  * [I am working on a new interpretability or attribution method and would like to add it to Captum. How do I proceed?](#i-am-working-on-a-new-interpretability-or-attribution-method-and-would-like-to-add-it-to-captum-how-do-i-proceed)
  * [I am using a gradient-based attribution algorithm such as integrated gradients for a RNN or LSTM network and I see 'cudnn RNN backward can only be called in training mode'. How can I resolve this issue ?](#how-can-I-resolve-cudnn-RNN-backward-error-for-RNN-or-LSTM-network)

* **[Optim](#optim)**

  * [How do I know if my model is compatible with the Optim?](#how-do-i-know-if-my-model-is-compatible-with-the-optim)
  * [Are only 3 channel RGB images and 4 channel RGBA images supported and can I use a different color space?](#are-only-3-channel-rgb-images-and-4-channel-rgba-images-supported-and-can-i-use-a-different-color-space)
  * [Why are my rendered visualizations poor quality or non-existent in outputs?](#why-are-my-rendered-visualizations-poor-quality-or-non-existent-in-outputs)
  * [Does the Optim module support JIT?](#does-the-optim-module-support-jit)
  * [What dtypes does the Optim module support?](#what-dtypes-does-the-optim-module-support)
  * [How can I avoid out of memory errors when rendering?](#how-can-i-avoid-out-of-memory-errors-when-rendering)
  * [Can I use Optim with a CPU or do I have to use a GPU?](#can-i-use-optim-with-a-cpu-or-do-i-have-to-use-a-gpu)
  * [Does Optim work with torch.fx?](#does-optim-work-with-torchfx)
  * [Does Optim support TensorBoard?](#does-optim-support-tensorboard)


## Attribution


### **How do I set the target parameter to an attribution method?**

The purpose of target is to select a single (scalar) value for each example in the output of your model to compute attributions based on the given target parameter. Each attribution method answers the question of how important each input value is towards a particular output scalar value.

If the model only returns a scalar value per example, which is common in either a regression case or binary classification, then you don’t need to pass the target argument or simply set it None.

If the model has a 2D output, which is common with classification cases, then you must pass a target to identify which output value to compute attributions for. Consider an output with the shape N x 3, with N being the number of examples and 3 being the number of classes. The purpose of targets is to select one of the 3 values for each example. You could then pass targets as just a single integer 0 or 1 or 2, which corresponds to attributing to the output for class 0, 1, or 2 respectively for all examples. If you are passing a batch of multiple examples (e.g. N = 4), you can also pass a different target for each example, by providing targets as a list or tensor, e.g. [0, 1, 0, 0] or torch.tensor([0, 1, 0, 0]), which would return attributions for class 0 for the first, third and fourth input examples (how important each input value was for the prediction of class 0), and class 1 for the second one (how important each input value was for the prediction of class 1).

If your model has > 2D output, then you can pass a tuple containing the indices of the particular scalar value in the output tensor for which attributions are desired. For instance, with an output shape of N x 3 x 4 x 5, target should be a tuple such as (2, 3, 2), corresponding to the index for each dimension other than the first. Like the 2D case, a list of tuples can be provided for a batch of input examples. Note that in some cases with > 2D output it may be beneficial to attribute with respect to the sum of particular output values rather than each value independently. The attribution methods would then answer the question of how important each input value is to the sum of the chosen values. To accomplish this, it is necessary to wrap your function in a function which sums the desired output values and provide the wrapper as the forward_func for the attribution method.


### **I am facing Out-Of-Memory (OOM) errors when using Captum. How do I resolve this?**

When using methods such as Integrated Gradients, Conductance, Internal Influence or other algorithms with n_steps argument, the input is expanded n_steps times, which often becomes too large to fit in memory. To address this issue, you can either reduce n_steps, which may lead to lower-quality approximations, or use the internal_batch_size argument, which allows dividing the expanded input into batches which are processed sequentially. Note that using internal_batch_size will increase runtime since it runs multiple evaluations, so it is best to utilize the largest feasible batch size within memory constraints.

If you are using a perturbation-based attribution algorithm, try reducing perturbations_per_eval if it is set to a value greater than 1. This will limit the number of perturbations processed in one batch.

For other algorithms, it might be necessary to try a smaller input batch size.

### **I am using a perturbation based method, and attributions are taking too long to compute. How can I speed it up?**

For perturbation based methods, including Feature Ablation and Occlusion, each perturbation of the input batch is processed sequentially by default. If enough memory is available or the input batch size is small, it is often possible to process multiple perturbations in one batch, which can substantially speed up the performance of these algorithms. To enable this, simply set the perturbations_per_eval argument to the desired value.

If you have multiple GPUs machines available, you can also wrap your model with DataParallel or look into torch.distributed / DistributedDataParallel, these PyTorch features work with all Captum methods.

### **Are SmoothGrad or VarGrad supported in Captum?**

Yes! SmoothGrad and VarGrad are available through NoiseTunnel in Captum, which can be used with any attribution algorithm in Captum. More details on Noise Tunnel can be found in the documentation [here](https://captum.ai/api/noise_tunnel.html).

### **How do I use Captum with BERT models?**

We have a tutorial demonstrating usage of Integrated Gradients on BERT [here](https://captum.ai/tutorials/Bert_SQUAD_Interpret).

### **My model inputs or outputs token indices, and when using Captum I see errors relating to gradients, how do I resolve this?**

For NLP models that take token indices as inputs, we cannot take gradients with respect to indices. To apply gradient-based attribution methods, it is necessary to replace the embedding layer with an InterpretableEmbedding layer or use LayerIntegratedGradients to compute attribution with respect to the embedding output. Attribution can then be summed for all dimensions of the embedding to evaluate importance of each token / index. For examples of this process, take a look at the [IMDB](https://captum.ai/tutorials/IMDB_TorchText_Interpret) or [BERT](https://captum.ai/tutorials/Bert_SQUAD_Interpret) tutorials.

If the output of the model is a token index, such as an image captioning cases, it is necessary to attribute with respect to the token score or probability rather than the index. Make sure that the model returns this and use target to choose the appropriate scalar score to attribute with respect to.

### **Can my model using functional non-linearities (E.g. nn.functional.ReLU) or reused modules be used with Captum?**

Most methods will work fine with functional non-linearities and arbitrary operations. Some methods, which require placing hooks during back-propagation, including DeepLift, DeepLiftShap, Guided Backpropagation, and Deconvolution will not work appropriately with functional non-linearities and must use the corresponding module activation (e.g. torch.nn.ReLU) which should be initialized in the module constructor. For DeepLift, it is important to also not reuse modules in the forward function, since this can cause issues in the propagation of multipliers. Computing layer or neuron attribution with layer modules that are used multiple times generally computes attributions for the last execution of the module. For more information regarding these restrictions, refer to the API documentation for the specific method, including DeepLift, DeepLiftShap, Guided Backpropagation, and Deconvolution.

### **Do JIT models, DataParallel models, or DistributedDataParallel models work with Captum?**

Yes, we have support for all these model types. Note that JIT models do not yet support hooks, so any methods using hooks including layer and neuron attribution methods, DeepLift, Guided Backprop, and Deconvolution are not supported. DataParallel and DistributedDataParallel are supported with all model types.

### **I am working on a new interpretability or attribution method and would like to add it to Captum. How do I proceed?**

For interpretability methods created by the community, we have two methods of involvement:

1. Awesome List - We maintain a list of interesting external projects that focus on interpretability that may be useful for users looking for functionality beyond what’s available in Captum.
2. Inclusion in Captum - New attribution algorithms that fit the structure of Captum can be considered for contribution to the contrib package of algorithms in Captum.  We review proposals for new additions to the contrib package on a case-by-case basis and consider factors such as publication history, quantitative and qualitative evaluation, citations, etc.

We are still working out the logistics of setting these up and will update this with more information once it’s available.

### **How can I resolve cudnn RNN backward error for RNN or LSTM network?**
If your model is set in eval mode you might run into errors, such as `cudnn RNN backward can only be called in training mode`, when you try to perform backward pass on a RNN / LSTM model in a GPU environment.
CuDNN with RNN / LSTM doesn't support gradient computation in eval mode that's why we need to disable cudnn for RNN in eval mode.
To resolve the issue you can set`torch.backends.cudnn.enabled` flag to False - `torch.backends.cudnn.enabled=False`


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

### **Does Optim work with torch.fx?**

No, the Optim module's dynamic control flow is not supported by [torch.fx](https://pytorch.org/docs/stable/fx.html). Custom modules though may be able to have some level of support for ``torch.fx``.

This means that Torchvision's ``torchvision.models.feature_extraction`` [package](https://pytorch.org/vision/stable/feature_extraction.html) won't work as it requires ``torch.fx`` support. The Optim module's ``collect_activations`` function should be used instead.

### **Does Optim support TensorBoard?**

Yes, Optim supports [TensorBoard](https://www.tensorflow.org/tensorboard) via PyTorch's ``torch.utils.tensorboard`` [module](https://pytorch.org/docs/stable/tensorboard.html). You will have to add PyTorch's TensorBoard related functions to your code first before using TensorBoard.

