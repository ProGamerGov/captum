---
id: algorithms
title: Algorithm Descriptions
---

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
