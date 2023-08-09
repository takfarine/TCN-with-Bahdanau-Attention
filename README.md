# Temporal Convolutional Block with Bahdanau Attention

## Overview
This repository contains an implementation of a Temporal Convolutional Block (TCN) enhanced with the Bahdanau Attention mechanism. TCNs offer powerful sequence modeling capabilities due to their ability to process long sequences and achieve expansive receptive fields. Incorporating attention mechanisms ensures the model captures not just temporal dependencies but also emphasizes important parts of the sequence.

## Key Components

1. **Causal Convolution (`CausalDWConv1D`):** 
   - Guarantees the causality of convolution operations to maintain the temporal order of sequences.
   
2. **Batch Normalization and Activation:**
   - Stabilizes the model's activations.

3. **Efficient Channel Attention (`ECA`):**
   - Implements channel-wise attention, emphasizing specific channels based on input.

4. **Bahdanau Attention:**
   - Weighs different parts/timesteps of the sequence to allow the model to focus on specific sequence parts when producing an output.

5. **Pointwise Convolution:**
   - Adjusts the dimensionality of the output, acting as a feature transformation mechanism.

6. **Residual Connection:**
   - Counters the vanishing gradient problem common in deep networks, often resulting in faster convergence.

## Usage

To incorporate the TCN block with Bahdanau Attention into your architecture, instantiate and use it like any Keras layer:

```python
x = TCNBlock(filters, kernel_size, dilation_rate)(input_tensor)


