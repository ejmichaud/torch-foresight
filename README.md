```
 _______             _
|__   __|           | |                               _---~~(~~-_.
   | | ___  _ __ ___| |__ ______                    _{        )   )
   | |/ _ \| '__/ __| '_ \______|                 ,   ) -~~- ( ,-' )_
   | | (_) | | | (__| | | |                      (  `-,_..`., )-- '_,)
 __|_|\___/|_|  \___|_| |_|      _     _        ( ` _)  (  -~( -_ `,  }
|  ____|                (_)     | |   | |       (_-  _  ~_-~~~~`,  ,' )
| |__ ___  _ __ ___  ___ _  __ _| |__ | |_        `~ ->(    __;-,((()))
|  __/ _ \| '__/ _ \/ __| |/ _` | '_ \| __|             ~~~~ {_ -_(())
| | | (_) | | |  __/\__ \ | (_| | | | | |_                     `\  }
|_|  \___/|_|  \___||___/_|\__, |_| |_|\__|                      { }
                            __/ |
                           |___/
```
[![Documentation Status](https://readthedocs.org/projects/torch-foresight/badge/?version=latest)](https://torch-foresight.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/ejmichaud/torch-foresight.svg?branch=master)](https://travis-ci.org/ejmichaud/torch-foresight)

This package provides a collection of modules useful for characterizing and predicting the dynamics and performance of neural nets. These consist mostly of novel metrics, derived from fields like theoretical neuroscience and information theory, aimed at helping researchers to better understand how neural networks work. The repository is meant to advance a new "Science of AI" or "Science of Deep Learning" (see [neuralnet.science](https://neuralnet.science)). It currently includes modules for computing:

* [Effective Information](https://arxiv.org/abs/1907.03902)

With the following under development:
- [ ] [Gradient Noise Scale](https://openai.com/blog/science-of-ai/)
- [ ] [Information Bottleneck](https://arxiv.org/abs/1503.02406)

**[Check out the comprehensive documentation (click me!)](https://torch-foresight.readthedocs.io)**

## Installation

The package currently only supports Python 3 (3.5-3.7). Pytorch is required as a dependency. If pytorch is already installed, simply use:

```
pip install git+https://github.com/ejmichaud/torch-foresight.git
```

If you don't have pytorch, installing it with anaconda is recommended. An `environment.yml` has been provided. Use it like so:
```
conda env create -f environment.yml
```
This will create a conda environment called "foresight", and install pytorch and then this package. It can then be activated with `conda activate foresight`. A requirements.txt file has also been provided, if you'd like to use `pip install -r requirements.txt`, which will install pytorch via pip. 

## Usage:

Computing effective information:

```python
import foresight.ei as ei

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

... (define model, data loaders) ...

input = next(iter(data_loader))[0].to(device) # get a batch to run model on
EI = ei.ei(model, input=input, device=device)
```

Effective information may prove a useful metric in characterizing the learning (generalization) and overfitting phases of a neural network. Here is an example of how it evolves during the training of a single layer (no hidden layers) softmax network:

<p align="center">
<img width="75%" src="docs/figures/single-layer-softmax-graph.png">
</p>
