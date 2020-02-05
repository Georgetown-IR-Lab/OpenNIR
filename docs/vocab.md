## Vocabularies

### Overview

Vocabularies let you swap out the underlying text representation of models.

### Word Vectors

OpenNIR includes three ways to handle out-of-vocabularly terms:
 - `vocab=wordvec` throws an error for OOV.
 - `vocab=wordvec_unk` uses a single `UNK` token to represent all OOV.
 - `vocab=wordvec_hash` uses multiple `UNK` tokens, assigned via the hash value of the OOV term.

There are four sources of word vectors implemented:
 - `vocab.source=fasttext` [FastText](https://arxiv.org/abs/1607.04606) vectors. Variants: `wiki-news-300d-1M` `crawl-300d-2M`
 - `vocab.source=glove` [GloVE](https://nlp.stanford.edu/pubs/glove.pdf) vectors. Variants: `cc-42b-300d`, `cc-840b-300d`
 - `vocab.source=convknrm` Vectors from the [ConvKNRM](http://www.cs.cmu.edu/~./callan/Papers/wsdm18-zhuyun-dai.pdf) experiments. Variants: `knrm-bing` `knrm-sogou`, `convknrm-bing` `convknrm-sogou`
 - `vocab.source=bionlp` Trained on PubMED `vocab.variant=pubmed-pmc`

### Contextualized Vectors `config/bert`

OpenNIR has a [BERT](https://arxiv.org/abs/1810.04805) implementation to provide contextualized
representations. Out-of-vocabulary terms are handled by the WordPiece tokenizer (i.e., split into
subords). This vocabularly also outputs CLS representation, which can be a useful signal for ranking.

There are two encoding stragegies:
 - `vocab.encoding=joint` - the query and document are modeled in the same sequence
 - `vocab.encoding=sep` - the query and document are modeled independently

The pretrained model weights can be configured with `vocab.bert_base` (default `bert-base-uncased`).
This accepts any value supported by the HuggingFace transformers library for the BERT model (see [here](https://huggingface.co/transformers/v2.4.0/pretrained_models.html)),
and the following:
 - [SciBERT](https://arxiv.org/abs/1903.10676): scibert-scivocab-uncased, scibert-scivocab-cased, scibert-basevocab-uncased, scibert-basevocab-cased
 - [BioBERT](https://arxiv.org/abs/1901.08746): biobert-pubmed-pmc, biobert-pubmed, biobert-pmc
