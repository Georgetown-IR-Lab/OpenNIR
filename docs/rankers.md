## Rankers


### Overview

You can configure and update the neural ranking arechitecture using `ranker`s.


### Implemented Architectures

#### CEDR-[KNRM/PACRR/DRMM] `config/cedr/`

Implementation of CEDR for the DRMM model described in:
Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. CEDR: Contextualized
Embeddings for Document Ranking. In SIGIR. [link](https://arxiv.org/abs/1904.07094)
Should be used with a model first trained using Vanilla BERT.

#### ConvKNRM `config/conv_krnm`

Implementation of the ConvKNRM model from:
Zhuyun Dai, Chenyan Xiong, Jamie Callan, and Zhiyuan Liu. 2018. Convolutional Neural
Networks for Soft-Matching N-Grams in Ad-hoc Search. In WSDM. [link](http://www.cs.cmu.edu/~./callan/Papers/wsdm18-zhuyun-dai.pdf)

#### DRMM `ranker=drmm`

Implementation of the DRMM model from:
Jiafeng Guo, Yixing Fan, Qingyao Ai, and William Bruce Croft. 2016. A Deep Relevance
Matching Model for Ad-hoc Retrieval. In CIKM. [link](https://arxiv.org/abs/1711.08611)

#### DUET (local) `ranker=duetl`

Implementation of the local variant of the Duet model from:
Bhaskar Mitra, Fernando Diaz, and Nick Craswell. 2016. Learning to Match using Local and
Distributed Representations of Text for Web Search. In WWW. [link](https://arxiv.org/abs/1610.08136)

#### KNRM `ranker=knrm`

Implementation of the K-NRM model from:
Chenyan Xiong, Zhuyun Dai, James P. Callan, Zhiyuan Liu, and Russell Power. 2017. End-to-End
Neural Ad-hoc Ranking with Kernel Pooling. In SIGIR. [link](https://arxiv.org/abs/1706.06613)

#### MatchPyramid `ranker=matchpyramid`

Implementation of the MatchPyramid model for ranking from:
Liang Pang, Yanyan Lan, Jiafeng Guo, Jun Xu, and Xueqi Cheng. 2016. A Study of MatchPyramid
Models on Ad-hoc Retrieval. In NeuIR @ SIGIR. [link](https://arxiv.org/abs/1606.04648)

#### PACRR `ranker=pacrr`

Implementation of the PACRR model from:
Kai Hui, Andrew Yates, Klaus Berberich, and Gerard de Melo. 2017. PACRR: A Position-Aware
Neural IR Model for Relevance Matching. In EMNLP. [link](https://arxiv.org/abs/1704.03940)

Some features included from CO-PACRR (e.g., shuf):
Kai Hui, Andrew Yates, Klaus Berberich, and Gerard de Melo. 2018. Co-PACRR: A Context-Aware
Neural IR Model for Ad-hoc Retrieval. In WSDM. [link](https://arxiv.org/abs/1706.10192)

#### Trivial `config/trivial`

Trivial ranker, which just returns the initial ranking score. Used for comparisions against
neural ranking approaches.

Options allow the score to be inverted (neg), the individual query term scores to be summed by
the ranker itself (qsum), and to use the manual relevance assessment instead of the run score,
representing an optimal re-ranker (max).

#### Vanilla Transformer `config/vanilla_bert`

Implementation of the Vanilla BERT model from:
Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. CEDR: Contextualized
Embeddings for Document Ranking. In SIGIR. [link](https://arxiv.org/abs/1904.07094)
Should be used with a transformer vocab, e.g., BertVocab.


### MatchZoo Architectures

Implementations of rankers from the [MatchZoo-py](https://github.com/NTMC-Community/MatchZoo-py) library.

#### KNRM `ranker=mz_knrm`

MatchZoo implementation of the K-NRM model from:
Chenyan Xiong, Zhuyun Dai, James P. Callan, Zhiyuan Liu, and Russell Power. 2017. End-to-End
Neural Ad-hoc Ranking with Kernel Pooling. In SIGIR. [link](https://arxiv.org/abs/1706.06613)

#### ConvKNRM `mz_conv_knrm`

MatchZoo implementation of the ConvKNRM model from:
Zhuyun Dai, Chenyan Xiong, Jamie Callan, and Zhiyuan Liu. 2018. Convolutional Neural
Networks for Soft-Matching N-Grams in Ad-hoc Search. In WSDM. [link](http://www.cs.cmu.edu/~./callan/Papers/wsdm18-zhuyun-dai.pdf)
