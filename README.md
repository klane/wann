# WekaNet: Weka Artificial Neural Network

[![Test Status](https://github.com/klane/wekanet/workflows/Tests/badge.svg)](https://github.com/klane/wekanet/actions)
[![Coverage Status](https://img.shields.io/codecov/c/github/klane/wekanet.svg?label=Coverage&logo=codecov)](https://codecov.io/gh/klane/wekanet)
[![License](https://img.shields.io/github/license/klane/wekanet.svg?label=License)](LICENSE)

WekaNet contains an artificial neural network framework for use with [Weka](http://www.cs.waikato.ac.nz/ml/weka/index.html) that provides more
control of the network structure than what is available in Weka's `MultilayerPerceptron` classifier.

Requires JDK 8+

## Features

- Networks, layers, and neurons are created with builders, so they are never in an intermediary state.
- All neurons in adjacent layers do not need to be connected (as in a multilayer perceptron), though this is the default behavior.
- Each neuron can have its own input and activation functions, including user-defined functions.
