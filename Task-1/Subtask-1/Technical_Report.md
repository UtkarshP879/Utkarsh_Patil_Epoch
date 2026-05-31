# Methodology:
The array element ranking task has been explored using
1. A Bidirectional LSTM (A Baseline Model)
2. An Encoder-Only Transformer with two separate architectures for data representation using:
   
   A. Continuous Representations
   
   B. Categorical Embeddings
3. An Encoder-Only Transformer without any Positional Encodings, following the continuous representation architecture

For the LSTM, as well as the transformer architectures with continuous data representation, the integer sequences have first been normalized using z-score normalization (in a sequence-wise manner), providing float inputs to the models. For the categorical embeddings architecture, the sequences have not been normalized and have only been input as the long datatype (integers).

All the models have been trained using the PyTorch framework with the Adam optimizer, and using the Cross-Entropy Loss.
After training and validating the models, their training and validation losses have been plotted, along with two types of validation accuracy metrics:
1. Token-wise Accuracy
   
2. Sequence-wise Accuracy

These metrics have also been reported for testing the models. After this, the attention weights for all the transformer models have been extracted and visualized as heatmaps.

# Architectural Choices:
## LSTM
A bidirectional LSTM has been used with a dropout of 0.3 within its fully-connected layer and with ReLU activation functions. The number of hidden dimensions are 64, decided after training with different numbers of hidden dimensions.
## Encoder-Only Transformer
The encoder architecture has been implemented exactly in accordance with the "Attention Is All You Need" paper, including the positional encodings calculation, the Q/K/V abstraction, the multi-head self-attention, the residual connections as well as the feed-forward networks. After experimenting with various values, the final hyperparameters of a 4-layer, 8-head, 128-dimensional transformer were determined.
## Transformer with Categorical Embeddings
Apart from the differences in the projection layers, (linear vs embeddings), the architecture of the transformer with categorical embeddings is identical to that of the original encoder architecture. However, in the feed-forward netwirk of each layer, a dropout of 0.3 has been incorporated to help with better generalization. It is a 4-layer, 8-head, 128-dimensional transformer.
## Transformer without Positional Encodings
Apart from the absence of positional encodings, the architecture of this transformer is identical to the original encoder architecture. It is a 4-layer, 8-head, 128-dimensional transformer.

# Numerical Representation Strategies:
A. Continuous Representation:

In this representation, the sequences were first normalized using z-score normalization and then input to the model as floats, where the model then used a linear layer to project the floats to the hidden-dimensional space. It allowed the LSTM as well as transformer architectures to converge smoothly. However, it also took more epochs for the models with this representation type to converge, because the model had to learn the ranking dependencies based off of the information within a single linear projection. However, as can be seen in the evaluation metrics section, this representation also led to the most accurate transformer models.

B. Categorical Embeddings:

In this representation, the sequences were not normalized. Instead, they were directly input as integers (specifically, the long datatype) to the transformer. Each token was then assigned its own unique embedding in the embedding table, which was then learnt as the training progressed. It can be seen from the loss plot of the transformer with categorical embeddings that it onverged much faster than the other transformer architectures, as well as the LSTM, since the embeddings were all separate and the model did not need to spend time figuring out the dependencies along a single linear projection. However, as a consequence of this, the model also became much less accurate, since it over-specialized the embeddings to the point where it is difficult for the model to infer the ranking relationship of that particular token in a sequencce upon which it has not been trained. Another major disadvantage of using this representation is the fact that the model cannot handle any out-of-distribution sequences, i.e., it does not have any embeddings for tokens (integers) that were not in its original dataset. This is not an issue seen in the other numerical representation strategy.

# Ablations and Experiments:
## 1. Using Categorical Embeddings:
