![Captum Logo](./website/static/img/captum_logo.png)

<hr/>

<!--- BADGES: START --->
[![GitHub - License](https://img.shields.io/github/license/pytorch/captum?logo=github&style=flat&color=green)][#github-license]
[![Conda](https://img.shields.io/conda/vn/pytorch/captum?logo=anaconda&style=flat&color=orange)](https://anaconda.org/pytorch/captum)
[![PyPI](https://img.shields.io/pypi/v/captum.svg)][#pypi-package]
[![CircleCI](https://circleci.com/gh/pytorch/captum.svg?style=shield)](https://circleci.com/gh/pytorch/captum)
[![Conda - Platform](https://img.shields.io/conda/pn/conda-forge/captum?logo=anaconda&style=flat)][#conda-forge-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/captum?logo=anaconda&style=flat&color=orange)][#conda-forge-package]
[![Conda Recipe](https://img.shields.io/static/v1?logo=conda-forge&style=flat&color=green&label=recipe&message=captum)][#conda-forge-feedstock]
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=captum&style=flat&color=pink&label=docs&message=captum)][#docs-package]

[#github-license]: https://github.com/pytorch/captum/blob/master/LICENSE
[#pypi-package]: https://pypi.org/project/captum/
[#conda-forge-package]: https://anaconda.org/conda-forge/captum
[#conda-forge-feedstock]: https://github.com/conda-forge/captum-feedstock
[#docs-package]: https://captum.ai/
<!--- BADGES: END --->


Captum is a model interpretability and understanding library for PyTorch.
Captum means comprehension in Latin and contains general purpose implementations
of integrated gradients, saliency maps, smoothgrad, vargrad and others for
PyTorch models. It has quick integration for models built with domain-specific
libraries such as torchvision, torchtext, and others.

*Captum is currently in beta and under active development!*


#### About Captum

With the increase in model complexity and the resulting lack of transparency, model interpretability methods have become increasingly important. Model understanding is both an active area of research as well as an area of focus for practical applications across industries using machine learning. Captum provides state-of-the-art algorithms, including Integrated Gradients, to provide researchers and developers with an easy way to understand which features are contributing to a model’s output.

For model developers, Captum can be used to improve and troubleshoot models by facilitating the identification of different features that contribute to a model’s output in order to design better models and troubleshoot unexpected model outputs.

Captum helps ML researchers more easily implement interpretability algorithms that can interact with PyTorch models. Captum also allows researchers to quickly benchmark their work against other existing algorithms available in the library.


#### Target Audience

The primary audiences for Captum are model developers who are looking to improve their models and understand which features are important and interpretability researchers focused on identifying algorithms that can better interpret many types of models.

Captum can also be used by application engineers who are using trained models in production. Captum provides easier troubleshooting through improved model interpretability, and the potential for delivering better explanations to end users on why they’re seeing a specific piece of content, such as a movie recommendation.

## Installation

**Installation Requirements**
- Python >= 3.6
- PyTorch >= 1.2


##### Installing the latest release

The latest release of Captum is easily installed either via
[Anaconda](https://www.anaconda.com/distribution/#download-section) (recommended) or via `pip`.

**with `conda`**

You can install captum from any of the following supported conda channels:

- channel: `pytorch`

  ```sh
  conda install captum -c pytorch
  ```

- channel: `conda-forge`

  ```sh
  conda install captum -c conda-forge
  ```

**With `pip`**

```bash
pip install captum
```

**Manual / Dev install**

If you'd like to try our bleeding edge features (and don't mind potentially
running into the occasional bug here or there), you can install the latest
master directly from GitHub. For a basic install, run:
```bash
git clone https://github.com/pytorch/captum.git
cd captum
pip install -e .
```

To customize the installation, you can also run the following variants of the
above:
* `pip install -e .[dev]`: Also installs all tools necessary for development
  (testing, linting, docs building; see [Contributing](#contributing) below).
* `pip install -e .[tutorials]`: Also installs all packages necessary for running the tutorial notebooks.

To execute unit tests from a manual install, run:
```bash
# running a single unit test
python -m unittest -v tests.attr.test_saliency
# running all unit tests
pytest -ra
```

## Getting Started

Please see the provided tutorials for more information.

## License
Captum is BSD licensed, as found in the [LICENSE](LICENSE) file.
