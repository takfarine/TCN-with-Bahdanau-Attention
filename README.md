<h4><center>:sunglasses:Temporal Convolutional Network (TCN) with Bahdanau Attention:sunglasses:</center></h4>

## Introduction:

This code provides an implementation of a Temporal Convolutional Block (TCN) integrated with the Bahdanau Attention mechanism. TCNs are powerful tools for sequence modeling tasks due to their ability to handle long sequences and have an expansive receptive field. When combined with the attention mechanism, this model not only captures temporal dependencies but also weighs the importance of different parts of the sequence.

#### Components:
* Causal Convolution (via CausalDWConv1D):
        Ensures that the convolution operations are causal, i.e., does not violate the temporal order of sequences.

* Batch Normalization and Activation:
        Helps in stabilizing the activations of the model.

* Efficient Channel Attention (via ECA):
        Provides a channel-wise attention mechanism, highlighting specific channels based on the input.

* Bahdanau Attention:
        Weighs the importance of different parts/timesteps of the sequence. It enables the model to focus on specific parts of the input sequence when producing an output.

* Pointwise Convolution:
        Used for changing the dimensionality of the output, acting as a feature transformation.

*  Residual Connection:
        Helps in avoiding the vanishing gradient problem, especially in deep networks, and often leads to faster convergence.
