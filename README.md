# OpenNIR
An end-to-end neural ad-hoc ranking pipeline.

## Quick start

*OpenNIR requires Python 3.6* (not tested with other versions).

Install dependencies

```bash
pip install -r requirements.txt
```

Train and validate a model (here, ConvKNRM on ANTIQUE):

```bash
scripts/pipeline.sh config/conv_knrm config/antique
```

(Performance on the test set can be obtained by adding `pipeline.test=True`)

Grid serach for BM25 over ANTIQUE for comparision with neural model performance:

```bash
scripts/pipeline.sh config/grid_search config/antique
```

(Performance on the test set can be obtained by adding `pipeline.test=True`)

Models, datasets, and vocabularies will be saved in `~/data/onir/`. This can be overridden by
setting `data_dir=~/some/other/place/` as a command line argument, in a configuration file, or in
the `ONIR_ARGS` environment variable.


## Features

### Rankers

 - DRMM `ranker=drmm` [paper](https://arxiv.org/abs/1711.08611)
 - Duet (local model) `ranker=duetl` [paper](https://arxiv.org/abs/1610.08136)
 - MatchPyramid `ranker=matchpyramid` [paper](https://arxiv.org/abs/1606.04648)
 - KNRM `ranker=knrm` [paper](https://arxiv.org/abs/1706.06613)
 - PACRR `ranker=pacrr` [paper](https://arxiv.org/abs/1704.03940)
 - ConvKNRM `ranker=conv_knrm` [paper](https://www.semanticscholar.org/paper/432b36c1bec275c2778c66f9897f9e02f7d8b579)
 - Vanilla BERT `config/vanilla_bert` [paper](https://arxiv.org/abs/1810.04805)
 - CEDR models `config/cedr/[model]` [paper](https://arxiv.org/abs/1810.04805)
 - MatchZoo models [source](https://github.com/NTMC-Community/MatchZoo)
   - MatchZoo's KNRM `ranker=mz_knrm`
   - MatchZoo's ConvKNRM `ranker=mz_conv_knrm`

### Datasets

 - TREC Robust 2004 `config/robust/fold[x]`
 - MS-MARCO `config/msmarco`
 - ANTIQUE `config/antique`
 - TREC CAR `config/car`
 - New York Times `config/nyt` -- for [content-based weak supervision](https://arxiv.org/abs/1707.00189)
 - TREC Arabic, Mandarin, and Spanish `config/multiling/*` -- for [zero-shot multilingual transfer learning](https://arxiv.org/pdf/1912.13080.pdf) ([instructions](https://opennir.net/multilingual.html))

### Evaluation Metrics

 - `map` (from trec_eval)
 - `ndcg` (from trec_eval)
 - `ndcg@X` (from trec_eval, gdeval)
 - `p@X` (from trec_eval)
 - `err@X` (from gdeval)
 - `mrr` (from trec_eval)
 - `rprec` (from trec_eval)
 - `judged@X` (implemented in python)

### Vocabularies

 - Binary term matching `vocab=binary` (i.e., changes interaction matrix from cosine similarity to to binary indicators)
 - Pretrained word vectors `vocab=wordvec`
   - `vocab.source=fasttext`
     - `vocab.variant=wiki-news-300d-1M`, `vocab.variant=crawl-300d-2M`
     - (information about FastText variants can be found [here](https://fasttext.cc/docs/en/english-vectors.html))
   - `vocab=source=glove`
   	 - `vocab.variant=cc-42b-300d`, `vocab.variant=cc-840b-300d`
   	 - (information about GloVe variants can be found [here](https://nlp.stanford.edu/projects/glove/))
   - `vocab.source=convknrm`
     - `vocab.variant=knrm-bing` `vocab.variant=knrm-sogou`, `vocab.variant=convknrm-bing` `vocab.variant=convknrm-sogou`
     - (information about ConvKNRM word embedding variants can be found [here](http://boston.lti.cs.cmu.edu/appendices/WSDM2018-ConvKNRM))
   - `vocab.source=bionlp`
     - `vocab.variant=pubmed-pmc`
     - (information about BioNLP variants can be found [here](http://bio.nlplab.org/))
 - Pretrained word vectors w/ single UNK vector for unknown terms `vocab=wordvec_unk`
   - (with above word embedding sources)
 - Pretrained word vectors w/ hash-based random selection for unknown terms `vocab=wordvec_hash` (defualt)
   - (with above word embedding sources)
 - BERT contextualized embeddings `vocab=bert`
   - Core models (from [HuggingFace](https://huggingface.co/pytorch-transformers/pretrained_models.html)): `vocab.bert_base=bert-base-uncased` (default), `vocab.bert_base=bert-large-uncased`, `vocab.bert_base=bert-base-cased`, `vocab.bert_base=bert-large-cased`, `vocab.bert_base=bert-base-multilingual-uncased`, `vocab.bert_base=bert-base-multilingual-cased`, `vocab.bert_base=bert-base-chinese`, `vocab.bert_base=bert-base-german-cased`, `vocab.bert_base=bert-large-uncased-whole-word-masking`, `vocab.bert_base=bert-large-cased-whole-word-masking`, `vocab.bert_base=bert-large-uncased-whole-word-masking-finetuned-squad`, `vocab.bert_base=bert-large-cased-whole-word-masking-finetuned-squad`, `vocab.bert_base=bert-base-cased-finetuned-mrpc`
   - [SciBERT](https://github.com/allenai/scibert): `vocab.bert_base=scibert-scivocab-uncased`, `vocab.bert_base=scibert-scivocab-cased`, `vocab.bert_base=scibert-basevocab-uncased`, `vocab.bert_base=scibert-basevocab-cased`
   - [BioBERT](https://github.com/dmis-lab/biobert) `vocab.bert_base=biobert-pubmed-pmc`, `vocab.bert_base=biobert-pubmed`, `vocab.bert_base=biobert-pmc`

## Citing OpenNIR

If you use OpenNIR, please cite the following WSDM demonstration paper:

```
@InProceedings{macavaney:wsdm2020-onir,
  author = {MacAvaney, Sean},
  title = {{OpenNIR}: A Complete Neural Ad-Hoc Ranking Pipeline},
  booktitle = {{WSDM} 2020},
  year = {2020}
}
```

## Acknowledgements

I gratefully acknowledge support for this work from the ARCS Endowment Fellowship. I thank Andrew
Yates, Arman Cohan, Luca Soldaini, Nazli Goharian, and Ophir Frieder for valuable feedback on the
manuscript and/or code contributions to OpenNIR.
