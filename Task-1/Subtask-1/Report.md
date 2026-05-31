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
A bidirectional LSTM has been used with a dropout of 0.3 within its fully-connected layer and ReLU activation functions. The number of hidden dimensions are 64, decided after training with different numbers of hidden dimensions.
## Encoder-Only Transformer
The encoder architecture has been implemented exactly in accordance with the "Attention Is All You Need" paper, including the positional encodings calculation, the Q/K/V abstraction, the multi-head self-attention, the residual connections as well as the feed-forward networks. After experimenting with various values, the final hyperparameters of a 4-layer, 8-head, 128-dimensional transformer were determined.
