# Previous/Related Work

## General Transformer Stuff

### [Peter Bloem's Blog Post on Transformers](http://peterbloem.nl/blog/transformers)

### [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)


## Transformers for Music

### [Magenta's Music Transformer](https://magenta.tensorflow.org/music-transformer)
   -> corresponding [paper](https://arxiv.org/pdf/1809.04281v3.pdf) <br>
   -> corresponding [PapersWithCode site](https://paperswithcode.com/paper/music-transformer)




## Training Methods

### [Post on the Wasserstein GAN Algorithm (WGAN)](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)
Original, naive implementation inherently minimises a Kullback-Leibler (or Jensen-Shannon) divergence. The Wasserstein exchanges that for the Wasserstein
(a.k.a. Earthmover) distance to address some problems such as mode collapse. Algorithm seems a simple enough to implement.





## Misc. and Less Relevant Stuff


 - [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf): Paper which discusses the problem with how BERT (and related models) is used to obtain embeddings for sentences (from words), the common methods being using the `[CLS}` token or averaging the word embeddings. Paper argues that BERT needs to be tuned in order to provide useful sentence embeddings. The proposed approach explicitly trains BERT to output useful cosine similarities for pairs of sentences (e.g. to reflect polarity or entailment between them).
