Overview
=================

About
-----------------

The Optim module is a set of tools for optimization based interpretability for neural networks. It is a continuation of the research work performed by the team behind the [tensorflow/lucid](https://github.com/tensorflow/lucid) library.

The Optim module is designed to be extremely customizable, as to avoid limitations in its research potential.

The initial concept for the Optim module was devised by Ludwig Shubert, and then developed by Ben Egan and Swee Kiat Lim with help from Chris Olah & Narine Kokhlikyan.


Structure
-----------------

![](https://user-images.githubusercontent.com/10626398/177629584-33e7ff7c-a504-404e-a7ab-d8d786b7e25a.svg?sanitize=true)

The standard rendering process works like this for the forward pass, with loss objectives being able to target any of the steps:

* ``NaturalImage`` (``ImageParameterization`` ➔ ``ToRGB`` ➔ Squash Function ➔ ``ImageTensor``) ➔ Transforms ➔ Model


Parameterizations
-----------------

The default settings store image parameters in a fully decorrelated format where the spatial information and channel information is decorrelated. By preconditioning our optimizer with decorrelated data, we alter the loss landscape to make optimization significantly easier and decrease the presence of high frequency patterns. Parameterizations like these are also known as differentiable image parameterizations.

![](https://user-images.githubusercontent.com/10626398/176753493-b90f4e18-0133-4dca-afd4-26e811aa965e.svg?sanitize=true)

* Decorrelated Data ➔ Recorrelate Spatial ➔ Recorrelate Color ➔ Squash Function ➔ Transforms ➔ Model

By default, recorrelation occurs entirely within the ``NaturalImage`` class.


Modules
-----------------

### Rendering Modules

**Images**: A torch.Tensor subclass for image tensors, and mutiple image parameterizations.

**Transforms**: The transforms module contains various transforms that are useful for performing transform robustness when rendering.

**Loss**: The loss module contains numerous loss objectives that are all fully composable with mathematical ops.

**Optimization**: The optimization module contains the optimization class and a stop criteria function.


### Submodules

**Reducer**: The reducer module makes it easy to perform dimensionality reduction with a wide array of algorithms like [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), [UMAP](https://umap-learn.readthedocs.io/en/latest/), [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), & [NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html).

**Circuits**: The circuits module allows for the extraction of meaningful weight interactions from between neurons which aren’t literally adjacent in a neural network.

**Models**: The models module contains the model zoo of pretrained models along with various help functions and classes.

**Dataset**: The dataset module provides functions for calculating color correlation matrices of image datasets.


Getting Started
-----------------

Below we demonstrate the usage of the Optim library for rendering a simple loss objective.

```
import torch
import captum.optim as opt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

We load the pre-trained InceptionV1 model instance.

```
model = opt.models.googlenet(pretrained=True).to(device)
```

Next we define our optimization objective as a layer channel target.

```
loss_fn = opt.loss.ChannelActivation(model.mixed4a, 465)
```

We also define a decorrelated image parameterization & a set of transforms.

```
image = opt.images.NaturalImage((224, 224)).to(device)
transforms = opt.transforms.TransformationRobustness()
```

We can now render the visualization using a simple helper function.

```
def visualize(model, loss_fn, image, transforms):
    obj = opt.InputOptimization(model, loss_fn, image, transforms)
    history = obj.optimize()
    image().show()

visualize(model, loss_fn, image, transforms)
```

**Circuits**

We start off by loading a linear version of the InceptionV1 model along with the normal version for rendering.

```
import captum.optim as opt
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load InceptionV1 model with nonlinear layers replaced by their linear equivalents
linear_model = (
    opt.models.googlenet(pretrained=True, use_linear_modules_only=True)
    .to(device)
    .eval()
)
model = opt.models.googlenet(pretrained=True).to(device).eval()


def visualize(model, loss_fn, image):
    transforms = opt.transforms.TransformationRobustness()
    obj = opt.InputOptimization(model, loss_fn, image, transforms)
    history = obj.optimize(opt.optimization.n_steps(256, True), lr=0.024)
    return image().detach()
```

```
# Extract expanded weights
W_4a_4b = opt.circuits.extract_expanded_weights(
    linear_model, linear_model.mixed4a_relu, linear_model.mixed4b_relu, 5
)

# Create heatmap image
W_4a_4b_hm = opt.weights_to_heatmap_2d(W_4a_4b[443, 308, ...] / W_4a_4b[443, ...].max())
hm_img = F.interpolate(W_4a_4b_hm[None, :], size=(224, 224), mode="nearest-exact")
```

From analysing the neurons in our model, we know that the mixed4a_relu channel 308 neuron is a dog head detector, and the mixed4b_relu channel 443 neuron is a full dog body. Viewing the weights connecting both neurons allows us to see the nessecary contextual information for how the head is placed on the body.

```
image = opt.images.NaturalImage((224, 224), batch=2).to(device)
loss_fn = opt.loss.NeuronActivation(model.mixed4a, 308, batch_index=0)
loss_fn += opt.loss.NeuronActivation(model.mixed4b_relu, 443, batch_index=1)
img = visualize(model, loss_fn, image)
img_set = torch.cat([img[0:1], hm_img, img[1:2]])

opt.show(img_set, images_per_row=3, figsize=(15, 10))
```

![circuits_showcase](https://user-images.githubusercontent.com/10626398/180571535-811fb987-d8c9-496f-9903-db52797930f6.png)


Docs
-----------------

The docs for the optim module can be found [here](https://captum.ai/api/).


Tutorials
-----------------

We also provide multiple tutorials covering a wide array of research for the optim module on the Captum website [here](https://captum.ai/tutorials/), or in the code repository [here](https://github.com/pytorch/captum/tree/master/tutorials).


FAQ
-----------------

### **How do I know if my model is compatible with the Optim?**

In general model layers need to be [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)s as functional layers don't support hooks. Please check out the 'Getting Started Model Preparation' tutorial notebook for more information.

### **Are only 3 channel RGB images and 4 channel RGBA images supported or can I use a different color space?**

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

If you are getting out of memory (OOM) errors when trying to render visualizations, you may have to reduce the batch size and or size of the image parameterization being used. If you are using a custom module, then you should make sure that there are no memory leaks present in it.

### **Does Optim work with torch.fx?**

No, the Optim module's dynamic control flow is not supported by [torch.fx](https://pytorch.org/docs/stable/fx.html). Custom modules though may be able to have some level of support for ``torch.fx``.

This means that Torchvision's ``torchvision.models.feature_extraction`` [package](https://pytorch.org/vision/stable/feature_extraction.html) won't work as it requires ``torch.fx`` support. The Optim module's ``collect_activations`` function should be used instead.

### **Does Optim support TensorBoard?**

Yes, Optim supports [TensorBoard](https://www.tensorflow.org/tensorboard) via PyTorch's ``torch.utils.tensorboard`` [module](https://pytorch.org/docs/stable/tensorboard.html). You will have to add PyTorch's TensorBoard related functions to your code first before using TensorBoard.


References
-----------------

* Color information for region segmentation: https://www.sciencedirect.com/science/article/pii/0146664X80900477

* Going Deeper with Convolutions: https://arxiv.org/abs/1409.4842

* Understanding Deep Image Representations by Inverting Them: https://arxiv.org/abs/1412.0035

* Inceptionism: Going Deeper into Neural Networks: https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html

* Visualizing Representations: Deep Learning and Human Beings: https://colah.github.io/posts/2015-01-Visualizing-Representations/

* Places: An Image Database for Deep Scene Understanding: https://arxiv.org/abs/1610.02055

* Using Artificial Intelligence to Augment Human Intelligence: https://distill.pub/2017/aia/

* Lucid: https://github.com/tensorflow/lucid

* Feature Visualization: https://distill.pub/2017/feature-visualization/

* Differentiable Image Parameterizations: https://distill.pub/2018/differentiable-parameterizations/

* The Building Blocks of Interpretability: https://distill.pub/2018/building-blocks/

* Exploring Neural Networks with Activation Atlases: https://distill.pub/2019/activation-atlas/

* Thread: Circuits: https://distill.pub/2020/circuits/

  * Visualizing Weights: https://distill.pub/2020/circuits/visualizing-weights/

  * Weight Banding: https://distill.pub/2020/circuits/weight-banding/

  * High-Low Frequency Detectors: https://distill.pub/2020/circuits/frequency-edges/

  * Curve Detectors: https://distill.pub/2020/circuits/curve-detectors/

* Multimodal Neurons in Artificial Neural Networks: https://distill.pub/2021/multimodal-neurons/
