# Vale's Developers Notebooks

Notebooks to mess around with model architectures and techniques.

# This has already been tried (and works):

  - using the TransformerEncoder to learn useful (see below) per-item embeddings of sequences
  - putting multiple task-specific "heads" on top of the encoder (i.e. the heads act as decoder), such as predicting the sum or whether a sequence is sorted <br>
    => super nice: the combination of encoder-decoder learn successfully *even if* the decoder is not trained at all (only randomly initialised); 
       the encoder learns to create useful projections **even for multiple tasks simultaneously**
  - a vanilla GAN training framework, with a simple, *untrained* feed-forward network as the sequence reconstruction decoder on top on the TransformerEncoder and a simple RNN as the adversary)
    => hyper-parameter tuning for successful learning is complex and unstable but the desired loss function graphs have been observed once or twice
    => in this attempt, the TransformerEncoder simultaneously had to learn to sum (see item above) and reconstruct vectors and was somewhat successful at both tasks
       implying that it can learn to provide embeddings that solve tasks (e.g. summation) while retaining a maximum amount of information about the original sequence
       
       
       
