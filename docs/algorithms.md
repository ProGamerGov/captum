---
id: algorithms
title: Algorithm Descriptions
---

Captum is a library within which different interpretability methods can be implemented.  The Captum team welcomes any contributions in the form of algorithms, methods or library extensions!


# Attribution

The attribution algorithms in Captum are separated into three groups, primary attribution, layer attribution and neuron attribution, which are defined as follows:
* Primary Attribution: Evaluates contribution of each input feature to the output of a model.
* Layer Attribution: Evaluates contribution of each neuron in a given layer to the output of the model.
* Neuron Attribution: Evaluates contribution of each input feature on the activation of a particular hidden neuron.

Below is a short summary of the various methods currently implemented for primary, layer, and neuron attribution within Captum, as well as noise tunnel, which can be used to smooth the results of any attribution method.

Beside attribution algorithms Captum also offers metrics to estimate the trustworthiness of model explanations.
Currently we offer infidelity and sensitivity metrics that help us to estimate the goodness of explanations.

## Primary Attribution
### Integrated Gradients
Integrated gradients represents the integral of gradients with respect to inputs along the path from a given baseline to input. The integral can be approximated using a Riemann Sum or Gauss Legendre quadrature rule. Formally, it can be described as follows:

![IG_eq1](/img/IG_eq1.png)
*Integrated Gradients along the i - th dimension of input X. Alpha is the scaling coefficient. The equations are copied from the [original paper](https://arxiv.org/abs/1703.01365).*

The cornerstones of this approach are two fundamental axioms, namely sensitivity and implementation invariance. More information regarding these axioms can be found in the original paper.

To learn more about Integrated Gradients, visit the following resources:
- [Original paper](https://arxiv.org/abs/1703.01365)

### Gradient SHAP
Gradient SHAP is a gradient method to compute SHAP values, which are based on Shapley values proposed in cooperative game theory. Gradient SHAP adds Gaussian noise to each input sample multiple times, selects a random point along the path between baseline and input, and computes the gradient of outputs with respect to those selected random points. The final SHAP values represent the expected value of gradients * (inputs - baselines).

The computed attributions approximate SHAP values under the assumptions that the input features are independent and that the explanation model is linear between the inputs and given baselines.

To learn more about GradientSHAP, visit the following resources:
- [SHAP paper](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
- [Original Implementation](https://github.com/slundberg/shap/#deep-learning-example-with-gradientexplainer-tensorflowkeraspytorch-models)

### DeepLIFT
DeepLIFT is a back-propagation based approach that attributes a change to inputs based on the differences between the inputs and corresponding references (or baselines) for non-linear activations.  As such, DeepLIFT seeks to explain the difference in the output from reference in terms of the difference in inputs from reference.  DeepLIFT uses the concept of multipliers to "blame" specific neurons for the difference in output.  The definition of a multiplier is as follows (from [original paper](https://arxiv.org/abs/1704.02685)):
![deepLIFT_eq1](/img/deepLIFT_multipliers_eq1.png)
*x is the input neuron with a difference from reference Δx, and t is the target neuron with a difference from reference Δt. C is then the contribution of Δx to Δt.*

Like partial derivatives (gradients) used in back propagation, multipliers obey the Chain Rule. According to the formulations proposed in [this paper](https://openreview.net/pdf?id=Sy21R9JAW). DeepLIFT can be overwritten as the modified partial derivatives of output of non-linear activations with respect to their inputs.

Currently, we only support Rescale Rule of DeepLIFT Algorithms. RevealCancel Rule will be implemented in later releases.

To learn more about DeepLIFT, visit the following resources:
- [Original paper](https://arxiv.org/abs/1704.02685)
- [Explanatory videos attached to paper](https://www.youtube.com/playlist?list=PLJLjQOkqSRTP3cLB2cOOi_bQFw6KPGKML)
- [Towards Better Understanding of Gradient-Based Attribution Methods for Deep Neural Networks](https://openreview.net/pdf?id=Sy21R9JAW)

### DeepLIFT SHAP
DeepLIFT SHAP is a method extending DeepLIFT to approximate SHAP values, which are based on Shapley values proposed in cooperative game theory. DeepLIFT SHAP takes a distribution of baselines and computes the DeepLIFT attribution for each input-baseline pair and averages the resulting attributions per input example.

DeepLIFT's rules for non-linearities serve to linearize non-linear functions of the network, and the method approximates SHAP values for the linearized version of the network. The method also assumes that the input features are independent.

To learn more about DeepLIFT SHAP, visit the following resources:
- [SHAP paper](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)

### Saliency
Saliency is a simple approach for computing input attribution, returning the gradient of the output with respect to the input. This approach can be understood as taking a first-order Taylor expansion of the network at the input, and the gradients are simply the coefficients of each feature in the linear representation of the model. The absolute value of these coefficients can be taken to represent feature importance.

To learn more about Saliency, visit the following resources:
- [Original paper](https://arxiv.org/abs/1312.6034)

### Input X Gradient
Input X Gradient is an extension of the saliency approach, taking the gradients of the output with respect to the input and multiplying by the input feature values. One intuition for this approach considers a linear model; the gradients are simply the coefficients of each input, and the product of the input with a coefficient corresponds to the total contribution of the feature to the linear model's output.

### Guided Backpropagation and Deconvolution
Guided backpropagation and deconvolution compute the gradient of the target output with respect to the input, but backpropagation of ReLU functions is overridden so that only non-negative gradients are backpropagated. In guided backpropagation, the ReLU function is applied to the input gradients, and in deconvolution, the ReLU function is applied to the output gradients and directly backpropagated. Both approaches were proposed in the context of a convolutional network and are generally used for convolutional networks, although they can be applied generically.

To learn more about Guided Backpropagation, visit the following resources:
- [Original paper](https://arxiv.org/abs/1412.6806)

To learn more about Deconvolution, visit the following resources:
- [Original paper](https://arxiv.org/abs/1311.2901)
- [Salient Deconvolutional Networks](https://link.springer.com/chapter/10.1007/978-3-319-46466-4_8)

### Guided GradCAM
Guided GradCAM computes the element-wise product of [guided backpropagation](###Guided-Backpropagation) attributions with upsampled (layer) [GradCAM](###GradCAM) attributions. GradCAM attributions are computed
with respect to a given layer, and attributions are upsampled to match the input size.
This approach is designed for convolutional neural networks. The chosen layer is often the last convolutional layer in the network, but any layer that is spatially aligned with the input can be provided.


Guided GradCAM was proposed by the authors of GradCAM as a method to combine the high-resolution nature of Guided Backpropagation with the class-discriminative advantages of GradCAM, which has lower resolution due to upsampling from a convolutional layer.

To learn more about Guided GradCAM, visit the following resources:
- [Original paper](https://arxiv.org/abs/1610.02391)
- [Website](http://gradcam.cloudcv.org/)

### Feature Ablation
Feature ablation is a perturbation based approach to compute attribution, involving replacing each input feature with a given baseline / reference value (e.g. 0), and computing the difference in output. Input features can also be grouped and ablated together rather than individually.
This can be used in a variety of applications. For example, for images, one can group an entire segment or region and ablate it together, measuring the importance of the segment (feature group).


### Feature Permutation
Feature permutation is a perturbation based approach which takes each feature individually, randomly permutes the feature values within a batch and computes the change in output (or loss) as a result of this modification. Like feature ablation, input features can also be grouped and shuffled together rather than individually.
Note that unlike other algorithms in Captum, this algorithm only provides meaningful attributions when provided with a batch of multiple input examples, as opposed to other algorithms, where a single example is sufficient.

To learn more about Feature Permutation, visit the following resources:
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/feature-importance.html)

### Occlusion
Occlusion is a perturbation based approach to compute attribution, involving replacing each contiguous rectangular region with a given baseline / reference, and computing the difference in output. For features located in multiple regions (hyperrectangles), the corresponding output differences are averaged to compute the attribution for that feature. Occlusion is most useful in cases such as images, where pixels in a contiguous rectangular region are likely to be highly correlated.

To learn more about Occlusion (also called grey-box / sliding window method), visit the following resources:
- [Original paper](https://arxiv.org/abs/1311.2901)
- [DeepExplain Implementation](https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py)

### Shapley Value Sampling
Shapley value is an attribution method based on a concept from cooperative game theory. This method involves taking each permutation of the input features and adding them one-by-one to a given baseline.
The output difference after adding each feature corresponds to its contribution, and these differences are averaged over all permutations to obtain the attribution.

Since this method is extremely computationally intensive for larger numbers of features, we also implement Shapley Value Sampling, where we sample some random permutations and average the marginal contribution of features based on these permutations.
Like feature ablation, input features can also be grouped and added together rather than individually.

To learn more about Shapley Value Sampling, visit the following resources:
- [Original paper](https://www.sciencedirect.com/science/article/pii/S0305054808000804)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/shapley.html)

### Lime
Lime is an interpretability method that trains an interpretable surrogate model by sampling data points around a specified input example and using model evaluations at these points to train a simpler interpretable 'surrogate' model, such as a linear model.

We offer two implementation variants of this method, LimeBase and Lime.
 LimeBase provides a generic framework to train a surrogate interpretable model. This differs from most API of other attribution methods, since the method returns a representation of the interpretable model (e.g. coefficients of the linear model). On the other hand, Lime provides a more specific implementation than LimeBase in order to expose a consistent API with other perturbation-based algorithms.

To learn more about Lime, visit the following resources:
- [Original paper](https://arxiv.org/abs/1602.04938)

### KernelSHAP
Kernel SHAP is a method that uses the LIME framework to compute Shapley Values. Setting the loss function, weighting kernel and regularization terms appropriately in the LIME framework allows theoretically obtaining Shapley Values more efficiently than directly computing Shapley Values.

To learn more about KernelSHAP, visit the following resources:
- [Original paper](https://arxiv.org/abs/1705.07874)

## Layer Attribution
### Layer Conductance
Conductance combines the neuron activation with the partial derivatives of both the neuron with respect to the input and the output with respect to the neuron to build a more complete picture of neuron importance.

Conductance builds on Integrated Gradients (IG) by looking at the flow of IG attribution which occurs through the hidden neuron.  The formal definition of total conductance of a hidden neuron *y* (from the [original paper](https://arxiv.org/abs/1805.12233)) is as follows:
![conductance_eq1](/img/conductance_eq_1.png)

For more efficient computation of layer conductance, we use the idea presented in this [paper](https://arxiv.org/abs/1807.09946) to avoid computing the gradient of each neuron with respect to the input.

To learn more about Conductance, visit the following resources:
- [Original Paper](https://arxiv.org/abs/1805.12233)
- [Computationally Efficient Measures of Internal Neuron Importance](https://arxiv.org/abs/1807.09946)

### Internal Influence
Internal Influence approximates the integral of gradients with respect to a particular layer along the path from a baseline input to the given input. This method is similar to applying integrated gradients, integrating the gradient with respect to the layer (rather than the input).

To learn more about Internal Influence, visit the following resources:
- [Original Paper](https://arxiv.org/abs/1802.03788)

### Layer Activation
Layer Activation is a simple approach for computing layer attribution, returning the activation of each neuron in the identified layer.

### Layer Gradient X Activation
Layer Gradient X Activation is the analog of the Input X Gradient method for hidden layers in a network. It element-wise multiplies the layer's activation with the gradients of the target output with respect to the given layer.

### GradCAM

GradCAM is a layer attribution method designed for convolutional neural networks, and is usually applied to the last convolutional layer.
GradCAM computes the gradients of the target output with respect to the given layer, averages for each output channel (dimension 2 of output), and multiplies the average gradient for each channel by the
layer activations. The results are summed over all channels and a ReLU is applied to the output, returning only non-negative attributions.

This procedure sums over the second dimension (# of channels), so the output of GradCAM attributions will have a second dimension of 1, but all other dimensions will match that of the layer output.

Although GradCAM directly attributes the importance of different neurons in the target layer, GradCAM is often used as a general attribution method. To accomplish this, GradCAM attributions are upsampled and viewed as a mask to the input, since a convolutional layer output generally matches the input image spatially.

To learn more about GradCAM, visit the following resources:
- [Original paper](https://arxiv.org/abs/1610.02391)
- [Website](http://gradcam.cloudcv.org/)

### Layer Integrated Gradients
Layer integrated gradients represents the integral of gradients with respect to the layer inputs / outputs along the straight-line path from the layer activations at the given baseline to the layer activation at the input.

To learn more about Integrated Gradients, see this [section](###Integrated-Gradients) above.

### Layer GradientSHAP
Layer GradientSHAP is the analog of GradientSHAP for a particular layer. Layer GradientSHAP adds Gaussian noise to each input sample multiple times, selects a random point along the path between baseline and input, and computes the gradient of the output with respect to the identified layer. The final SHAP values approximate the expected value of gradients * (layer activation of inputs - layer activation of baselines).

To learn more about Gradient SHAP, see this [section](###Gradient-SHAP) above.

### Layer DeepLIFT
Layer DeepLIFT is the analog of the DeepLIFT method for hidden layers in a network.

To learn more about DeepLIFT, see this [section](###DeepLIFT) above.

### Layer DeepLIFT SHAP

Layer DeepLIFT SHAP is the analog of DeepLIFT SHAP for a particular layer. Layer DeepLIFT SHAP takes a distribution of baselines and computes the Layer DeepLIFT attribution for each input-baseline pair and averages the resulting attributions per input example.

To learn more about DeepLIFT SHAP, see this [section](###DeepLIFT-SHAP) above.

### Layer Feature Ablation
Layer feature ablation is the analog of feature ablation for an identified layer input or output. It is a perturbation based approach to compute attribution, involving replacing each value in the identified layer with a given baseline / reference value (e.g. 0), and computing the difference in output. Values within the layer can also be grouped and ablated together rather than individually.

## Neuron Attribution
### Neuron Conductance
Conductance combines the neuron activation with the partial derivatives of both the neuron with respect to the input and the output with respect to the neuron to build a more complete picture of neuron importance.

Conductance for a particular neuron builds on Integrated Gradients (IG) by looking at the flow of IG attribution from each input through the particular neuron.  The formal definition of conductance of neuron y for the attribution of input i (from the [original paper](https://arxiv.org/abs/1805.12233)) is as follows:
![conductance_eq2](/img/conductance_eq_2.png)

Note that based on this definition, summing the neuron conductance (over all input features) always equals the layer conductance for the particular neuron.

To learn more about Conductance, visit the following resources:
- [Original Paper](https://arxiv.org/abs/1805.12233)
- [Computationally Efficient Measures of Internal Neuron Importance](https://arxiv.org/abs/1807.09946)

### Neuron Gradient
Neuron gradient is the analog of the saliency method for a particular neuron in a network. It simply computes the gradient of the neuron output with respect to the model input. Like Saliency, this approach can be understood as taking a first-order Taylor expansion of the neuron's output at the given input, and the gradients correspond to the coefficients of each feature in the linear representation of the model.

### Neuron Integrated Gradients
Neuron Integrated Gradients approximates the integral of input gradients with respect to a particular neuron along the path from a baseline input to the given input. This method is equivalent to applying integrated gradients
      considering the output to be simply the output of the identified neuron.

To learn more about Integrated Gradients, see this [section](###Integrated-Gradients) above.

### Neuron Guided Backpropagation and Deconvolution
Neuron guided backpropagation and neuron deconvolution are the analogs of guided backpropagation and deconvolution for a particular neuron.

To learn more about Guided Backpropagation and Deconvolution, see this [section](###Guided-Backpropagation-and-Deconvolution) above.

### Neuron GradientSHAP
Neuron GradientSHAP is the analog of GradientSHAP for a particular neuron. Neuron GradientSHAP adds Gaussian noise to each input sample multiple times, selects a random point along the path between baseline and input, and computes the gradient of the target neuron with respect to each selected random points. The final SHAP values approximate the expected value of gradients * (inputs - baselines).

To learn more about GradientSHAP, see this [section](###Gradient-SHAP) above.

### Neuron DeepLIFT
Neuron DeepLIFT is the analog of the DeepLIFT method for a particular neuron.

To learn more about DeepLIFT, see this [section](###DeepLIFT) above.

### Neuron DeepLIFT SHAP

Neuron DeepLIFT SHAP is the analog of DeepLIFT SHAP for a particular neuron. Neuron DeepLIFT SHAP takes a distribution of baselines and computes the Neuron DeepLIFT attribution for each input-baseline pair and averages the resulting attributions per input example.

To learn more about DeepLIFT SHAP, see this [section](###DeepLIFT-SHAP) above.

### Neuron Feature Ablation
Neuron feature ablation is the analog of feature ablation for a particular neuron. It is a perturbation based approach to compute attribution, involving replacing each input feature with a given baseline / reference value (e.g. 0), and computing the difference in the target neuron's value. Input features can also be grouped and ablated together rather than individually.
This can be used in a variety of applications. For example, for images, one can group an entire segment or region and ablate it together, measuring the importance of the segment (feature group).

## Noise Tunnel
Noise Tunnel is a method that can be used on top of any of the attribution methods. Noise tunnel computes attribution multiple times, adding Gaussian noise to the input each time, and combines the calculated attributions based on the chosen type. The supported types for noise tunnel are:
* Smoothgrad: The mean of the sampled attributions is returned. This approximates smoothing the given attribution method with a Gaussian Kernel.
* Smoothgrad Squared: The mean of the squared sample attributions is returned.
* Vargrad: The variance of the sample attributions is returned.

To learn more about Noise Tunnel methods, visit the following resources:
- [SmoothGrad Original paper](https://arxiv.org/abs/1706.03825)
- [VarGrad Original paper](https://arxiv.org/abs/1810.03307)

## Metrics
### Infidelity
Infidelity measures the mean squared error between model explanations in the magnitudes of input perturbations and predictor function's changes to those input perturbtaions. Infidelity is defined as follows:
![infidelity_eq](/img/infidelity_eq.png)
It is derived from the completeness property of well-known attribution algorithms, such as Integrated Gradients, and is a computationally more efficient and generalized notion of Sensitivy-n. The latter measures correlations between the sum of the attributions and the differences of the predictor function at its input and fixed baseline. More details about the Sensitivity-n can be found here:
https://arxiv.org/abs/1711.06104
More details about infidelity measure can be found here:
- [Original paper](https://arxiv.org/abs/1901.09392)

### Sensitivity
Sensitivity measures the degree of explanation changes to subtle input perturbations using Monte Carlo sampling-based approximation and is defined
as follows:
![sensitivity_eq](/img/sensitivity_eq.png)
In order to approximate sensitivity measure, by default, we sample from a sub-space of an L-Infinity ball with a default radius.
The users can modify both the radius of the ball and the sampling function.
More details about sensitivity measure can be found here:
- [Original paper](https://arxiv.org/abs/1901.09392)


# Optim

Below is a quick summary of the loss objectives currently provided by Optim. Loss objectives are used to steer the optimization process towards desired directions, layers, channels, and neurons.

Loss objectives can be made to target specific batch indices, and they are fully composable with mathematical operations.

## Loss Objectives

### LayerActivation
This is the most basic loss available and it simply returns the activations in their original form.

* Pros: Can potentially give a broad overview of a target layer.
* Cons: Not specific enough for most research tasks.

### ChannelActivation
This loss maximizes the activations of a target channel in a specified target layer, and can be useful to determine what features the channel is excited by.

* Pros: A good balance between neuron and layer activation.
* Cons: Can be very polysemantic in many cases. Channels with high degrees of polysemanticity can be difficult to interpet.

### NeuronActivation:
This loss maximizes the activations of a target neuron in the specified channel from the specified layer. This loss is useful for determining the type of features that excite a neuron, and thus is often used for circuits and neuron related research.

* Pros: Extremely specific in what it targets, and thus the information obtained can be extremely useful.
* Cons: Sometimes you don’t want something overly specific. Neurons don’t scale well to larger image sizes when rendering.

To learn more about NeuronActivation visit the following resources:

* [Research: Neuron Mechanics](https://github.com/tensorflow/lucid/issues/110)
* [Original Implementation](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/objectives.py)

### DeepDream
This loss returns the squared layer activations. When combined with a negative mean loss summarization, this loss will create hallucinogenic visuals commonly referred to as ‘Deep Dream’ or ’DeepDream’. 

DeepDream tries to increase the values of neurons proportional to the amount they are presently active. This is equivalent to maximizing the sum of the squares. If you remove the square, you'd be visualizing a direction of: ``[1,1,1,....]``.

* Pros: Can create visually interesting images.
* Cons: Not specific enough for most research tasks.

To learn more about DeepDream visit the following resources:

* [Inceptionism: Going Deeper into Neural Networks](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
* https://github.com/google/deepdream
* https://en.wikipedia.org/wiki/DeepDream
* [Lucid Implementation](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/objectives.py)

### TotalVariation
This loss attempts to smooth / denoise the target by performing total variance denoising. The target is most often the image that’s being optimized. This loss is often used to remove unwanted visual artifacts.

* Pros: Can remove unwanted visual artifacts.
* Cons: Can result in less sharp / more blurry visualization images.

To learn more about TotalVariation visit the following resources:

* [Understanding Deep Image Representations by Inverting Them](https://arxiv.org/abs/1412.0035)
* https://en.wikipedia.org/wiki/Total_variation
* [Original Implementation](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/objectives.py)

### L1
Penalizes the l1 of the target layer activations.

* Pros: Can be used as a penalty, similar to L1 regularization.
* Cons:

To learn more about L1 visit the following resources:

* [Original Implementation](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/objectives.py)

### L2
Penalizes the l2 of the target layer activations.

* Pros: Can be used as a penalty, similar to L2 regularization.
* Cons:

To learn more about L2 visit the following resources:

* [Original Implementation](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/objectives.py)

### Diversity
This loss helps break up polysemantic layers, channels, and neurons by encouraging diversity across the different batches. This loss is to be used along with a main loss. Diversity is very similar to [neural style transfer](https://en.wikipedia.org/wiki/Neural_style_transfer), but style similarity is penalized instead of being encouraged.

* Pros: Helps separate polysemantic features into different images.
* Cons: Requires a batch size greater than 1, and can be extremely slow with large batch sizes. This loss really only works on targets that are polysemantic. There is also no control over how features are separated out into groups.

To learn more about Diversity visit the following resources:

* [Feature Visualization: Diversity](https://distill.pub/2017/feature-visualization/#diversity)
* [Research: Poly-Semantic Neurons](https://github.com/tensorflow/lucid/issues/122)
* [Original Implementation](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/objectives.py)

### ActivationInterpolation
This loss helps to interpolate or mix visualizations from two activations (layer or channel) by interpolating a linear sum between the two activations.

* Pros: Can create visually interesting images, especially when used with Alignment.
* Cons: Interpolations may not be semantically useful beyond visual interest.

To learn more about ActivationInterpolation visit the following resources:

* [Feature Visualization: Interaction between Neurons](https://distill.pub/2017/feature-visualization/#Interaction-between-Neurons)
* [Original Implementation](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/objectives.py)

### Alignment
When interpolating between activations, it may be desirable to keep image landmarks in the same position for visual comparison. This loss helps to minimize L2 distance between neighbouring images. 

* Pros: Helps to make interpolated images more comparable.
* Cons: Resulting images may be less semantically representative of the channel/layer/neuron, since we are forcing images to also be visually aligned.

To learn more about Alignment visit the following resources:

* [Feature Visualization: Interaction between Neurons](https://distill.pub/2017/feature-visualization/#Interaction-between-Neurons)
* [Research: "The Art of Dimensionality Reduction"](https://github.com/tensorflow/lucid/issues/111)
* [Original Implementation](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/objectives.py)

### Direction
This loss helps to visualize a specific vector direction in a layer, by maximizing the alignment between the input vector and the layer’s activation vector. The dimensionality of the vector should correspond to the number of channels in the layer.

* Pros: Szegedy et al. and Bau et al. respectively found that activations along random and basis directions could be semantically meaningful and this loss allows us to visualize these directions.
* Cons: Largely random and, as of now, no structured way to find meaningful directions.

To learn more about Direction visit the following resources:

* [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199)
* [Network Dissection: Quantifying Interpretability of Deep Visual Representations](https://arxiv.org/abs/1704.05796)
* [Research: Feature Visualization Objectives](https://github.com/tensorflow/lucid/issues/116)
* [Original Implementation](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/objectives.py)

### NeuronDirection
Extends Direction loss by focusing on visualizing a single neuron within the kernel.

* Pros: See Direction loss.
* Cons: See Direction loss.

To learn more about NeuronDirection visit the following resources:

* [Exploring Neural Networks with Activation Atlases](https://distill.pub/2019/activation-atlas/)
* [Research: Feature Visualization Objectives](https://github.com/tensorflow/lucid/issues/116)
* [Original Implementation](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/objectives.py)

### AngledNeuronDirection
This objective is similar to NeuronDirection, but it places more emphasis on the angle by optionally multiplying the dot product by the cosine similarity.

* Pros: More useful for visualizing activation atlas images.

To learn more about AngledNeuronDirection visit the following resources:

* [Exploring Neural Networks with Activation Atlases](https://distill.pub/2019/activation-atlas/)
* [Research: Feature Visualization Objectives](https://github.com/tensorflow/lucid/issues/116)
* [Original Implementation](https://github.com/tensorflow/lucid/blob/master/notebooks/activation-atlas/activation-atlas-simple.ipynb)

### TensorDirection
Extends Direction loss by allowing batch-wise direction visualization.

* Pros: See Direction loss.
* Cons: See Direction loss.

To learn more about TensorDirection visit the following resources:

* [Original Implementation](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/objectives.py)

### ActivationWeights
This loss weighs specific channels or neurons in a given layer, via a weight vector. 

* Pros: Allows for region and dimension specific weighting.
* Cons: Requires knowledge beforehand of the target region.

To learn more about ActivationWeights visit the following resources:

* [Original Implementation](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/objectives.py)

### L2Mean
A simple L2 penalty where the mean is used instead of the square root of the sum.

* Pros: It was found to work better for CLIP visualizations than the traditional L2 objective.
* Cons:

To learn more about L2Mean visit the following resources:

* [Multimodal Neurons in Artificial Neural Networks: Faceted Feature Visualization](https://distill.pub/2021/multimodal-neurons/#faceted-feature-visualization)
  * [Supplementary Implementation](https://github.com/openai/CLIP-featurevis/blob/master/example_facets.py)

### VectorLoss

This loss objective is similar to the Direction objective, except it computes the matrix product of the activations and vector, rather than the cosine similarity. In addition to optimizing towards channel directions, this objective can also perform a similar role to the ChannelActivation objective by using one-hot 1D vectors.

* Pros:
* Cons:

To learn more about VectorLoss visit the following resources:

* [Multimodal Neurons in Artificial Neural Networks: Faceted Feature Visualization](https://distill.pub/2021/multimodal-neurons/#faceted-feature-visualization)
  * [Supplementary Implementation](https://github.com/openai/CLIP-featurevis/blob/master/example_facets.py)

### FacetLoss

The FacetLoss objective allows us to steer feature visualization towards a particular theme / concept. This is done by using the weights from linear probes trained on the lower layers of a model to discriminate between a certain theme or concept and generic natural images.

* Pros: Works on highly polysemantic / highly faceted targets where the Diversity objective fails due to lack of specificity. Doesn't require a batch size greater than 1 to work.
* Cons: Requires training linear probes on the target layers using training images from the desired facet.

To learn more about FacetLoss visit the following resources:

* [Multimodal Neurons in Artificial Neural Networks: Faceted Feature Visualization](https://distill.pub/2021/multimodal-neurons/#faceted-feature-visualization)
  * [Supplementary Implementation](https://github.com/openai/CLIP-featurevis/blob/master/example_facets.py)
