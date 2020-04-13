# Teaching a New Dog Old Tricks: Resurrecting Multilingual Retrieval Using Zero-shot Learning

Below are the steps to reproduce the primary results from _Teaching a New Dog Old Tricks: Resurrecting
Multilingual Retrieval Using Zero-shot Learning.  Sean MacAvaney, Luca Soldaini, Nazli Goharian.
ECIR 2020 (short)_. [pdf](https://arxiv.org/pdf/1912.13080.pdf)

The code to reproduce the paper is incorporated into [OpenNIR](https://github.com/Georgetown-IR-Lab/OpenNIR).
Refer to https://opennir.net/ go get started.

Note that the precise values differ slightly from the numbers reported in the original paper (usually
these results are higher) due to improvements in [Anserini](https://github.com/castorini/Anserini) and
other dependent software packages since the experiments were originally run. We note that the trends
we observe are the same as reported in the paper.


## First Step

Before all else, you need to initialize the TREC Robust 2004 dataset:

```
$ scripts/init_dataset.sh dataset=robust
[snip]
dataset is initialized (528030 documents)
```


## Arabic

Start by initializing the dataset. You'll need a copy of [LDC2001T55](https://catalog.ldc.upenn.edu/LDC2001T55)
as the document collections, but all other files will be downloaded from TREC.

```
$ scripts/init_dataset.sh dataset=trec_arabic
[snip]
dataset is initialized (383743 documents)
```

Let's run BM25 as a baseline:

```
$ bash scripts/pipeline.sh config/trivial/bm25 config/multiling/arabic_2001
[snip]
map=0.3582 ndcg@20=0.6018 p@20=0.5420

$ bash scripts/pipeline.sh config/trivial/bm25 config/multiling/arabic_2002
[snip]
map=0.2925 ndcg@20=0.4066 p@20=0.3670
```

Now train a multi-lingual BERT model on TREC Robust, and evaluate it on TREC Aarbic:

```
$ bash scripts/pipeline.sh config/vanilla_bert config/multiling/arabic_2001
[snip]
map=0.3645 ndcg@20=0.6464 p@20=0.5840

$ bash scripts/pipeline.sh config/vanilla_bert config/multiling/arabic_2002
[snip]
map=0.3073 ndcg@20=0.4223 p@20=0.3830
```



## Mandarin

Start by initializing the dataset. You'll need a copy of [LDC2000T51](https://catalog.ldc.upenn.edu/LDC2000T51)
as the document collections, but all other files will be downloaded from TREC.

```
$ scripts/init_dataset.sh dataset=trec_mandarin
[snip]
dataset is initialized (164778 documents)
```

Let's run BM25 as a baseline:

```
$ bash scripts/pipeline.sh config/trivial/bm25 config/multiling/mandarin_5
[snip]
map=0.2953 ndcg@20=0.4125 p@20=0.3946

$ bash scripts/pipeline.sh config/trivial/bm25 config/multiling/mandarin_6
[snip]
map=0.3720 ndcg@20=0.6272 p@20=0.5885
```

Now train a multi-lingual BERT model on TREC Robust, and evaluate it on TREC Mandarin:

```
$ bash scripts/pipeline.sh config/vanilla_bert config/multiling/mandarin_5
[snip]
map=0.3490 ndcg@20=0.5256 p@20=0.5107

$ bash scripts/pipeline.sh config/vanilla_bert config/multiling/mandarin_6
[snip]
map=0.4093 ndcg@20=0.7169 p@20=0.6788
```



## Spanish

Start by initializing the dataset. You'll need a copy of [LDC2000T51](https://catalog.ldc.upenn.edu/LDC2000T51)
as the document collections, but all other files will be downloaded from TREC.

```
bash scripts/init_dataset.sh dataset=trec_spanish
[snip]
dataset is initialized (57868 documents)
```

Let's run BM25 as a baseline (note that TREC 4 only has description queries, so we use those there):

```
$ bash scripts/pipeline.sh config/trivial/bm25 config/multiling/spanish_3
[snip]
map=0.3425 ndcg@20=0.5149 p@20=0.5000

$ bash scripts/pipeline.sh config/trivial/bm25 config/multiling/spanish_4
[snip]
map=0.2099 ndcg@20=0.4197 p@20=0.3820
```

Now train a multi-lingual BERT model on TREC Robust, and evaluate it on TREC Spanish:

```
$ bash scripts/pipeline.sh config/vanilla_bert config/multiling/spanish_3
[snip]
map=0.3684 ndcg@20=0.6344 p@20=0.6200

$ bash scripts/pipeline.sh config/vanilla_bert config/multiling/spanish_4
[snip]
map=0.2158 ndcg@20=0.4780 p@20=0.4400
```



## Result summary

| Dataset           | BM25 P@20 | BERT P@20| BM25 nDCG@20 | BERT nDCG@20 | BM25 MAP | BERT MAP |
| :---------------  | --------: | --------:| -----------: | -----------: | -------: | -------: |
| TREC Arabic 2001  |    0.5420 |    0.5840|       0.6018 |       0.6464 |   0.3582 |   0.3645 |
| TREC Arabic 2002  |    0.3670 |    0.3830|       0.4066 |       0.4223 |   0.2925 |   0.3073 |
| TREC Mandarin 5   |    0.3946 |    0.5107|       0.4125 |       0.5256 |   0.2953 |   0.3490 |
| TREC Mandarin 6   |    0.5885 |    0.6788|       0.6272 |       0.7169 |   0.3720 |   0.4093 |
| TREC Spanish 3    |    0.5000 |    0.6200|       0.5149 |       0.6344 |   0.3425 |   0.3684 |
| TREC Spanish 4    |    0.3820 |    0.4400|       0.4197 |       0.4780 |   0.2099 |   0.2158 |



## Citation

If you use this work, please cite:

```
@inproceedings{macavaney:ecir2020-multiling,
  author = {MacAvaney, Sean and Soldaini, Luca and Goharian, Nazli},
  title = {Teaching a New Dog Old Tricks: Resurrecting Multilingual Retrieval Using Zero-shot Learning},
  booktitle = {ECIR},
  year = {2020}
}
```
