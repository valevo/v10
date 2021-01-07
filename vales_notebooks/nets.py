import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Embedding, Linear, Softmax, NLLLoss, RNN, Sigmoid

# TRANSFORMER ENCODER

class LM(torch.nn.Module):
    def __init__(self, vocab_dim, embed_dim, num_layers=1):
        super().__init__()
        
        self.vocab_dim = vocab_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.emb = Embedding(vocab_dim, self.embed_dim)
        
        # nhead needs to divide d_model (embedding dimension)
        self.encoder_layer = TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.embed_dim//2)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        
    def forward(self, x):
        x_emb = self.emb(x)
        out = self.encoder(x_emb)
        return out  # output shape is (x.shape[0], x.shape[1], self.embed_dim)
    

    
# BASE CLASS FOR DECODERS
class Head(torch.nn.Module):
    def __init__(self, lm, out_dim):
        super().__init__()
        self.lm = lm
        self.in_dim = lm.embed_dim
        self.out_dim = out_dim
        
        self.linear = Linear(in_features=self.in_dim, out_features=self.out_dim)
    
    def forward(self, lm_input):
        return self.linear(lm(lm_input))
    

# DECODER CLASS FOR USING AGGREGATION FUNCTIONS (WHERE OUTPUT DIM IS 1)
class AggregateHead(Head):
    @staticmethod
    def aggr_sum(lm_output):
        # lm output shape:(batch, seq_len, transformer_dim)
        return lm_output.sum(1)
    
    @staticmethod
    def aggr_mean(lm_output):
        # lm output shape:(batch, seq_len, transformer_dim)
        return lm_output.mean(1)
    
    def __init__(self, lm, aggregate_function):
        super().__init__(lm, out_dim=1)
        
        self.aggr_f = aggregate_function
        from torch.nn import Identity
        self.final_layer = Identity()
        
    def forward(self, lm_output):
#         lm_out = lm(lm_input)
        aggr = self.aggr_f(lm_output)
        return self.final_layer(self.linear(aggr))
    

class ReconstructHead(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.vocab_dim = lm.vocab_dim
        self.in_dim = lm.embed_dim
        self.linear = Linear(in_features=self.in_dim, out_features=self.vocab_dim)
        self.softmax = Softmax(dim=-1)
        
    def forward(self, lm_output):
#         encoded = self.lm(lm_input)
        return self.softmax(self.linear(lm_output))
    

# GAN
class GAN(torch.nn.Module):
    def __init__(self, vocab_dim, hidden_dim):
        super().__init__()
        self.h = hidden_dim
        self.embed = Embedding(vocab_dim, vocab_dim)
        self.rnn = RNN(input_size=vocab_dim, hidden_size=self.h,
                        num_layers=1, bidirectional=False,
                        batch_first=True, dropout=0.1)
        
        self.linear = Linear(in_features=hidden_dim, out_features=1)
        self.sig = Sigmoid()
        
    def forward(self, vectors):
        rnn_outs, hidden = self.rnn(self.embed(vectors))
        rnn_outs = rnn_outs.sum(dim=1)
        
        assert rnn_outs.shape == (vectors.shape[0], self.h)
        return self.sig(self.linear(rnn_outs))
        
        
    def mean_prob(self, vectors):
        # loss is the fraction of supplied vectors which the GAN judges to be NOT 'real'
        # or actually the average probability that the GAN assigns
        judgements = self.forward(vectors)
        return judgements.mean()
    
    def loss(self, vectors):
        return 1 - self.mean_prob(vectors)