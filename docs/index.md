## OpenNIR

_A Complete Neural Ad-Hoc Ranking Pipeline_

From [Georgetown Information Retrieval Lab](http://ir.cs.georgetown.edu/)


### Key Features

#### An end-to-end [neural ranking pipeline](pipeline.html)

 - Implementations of numerous [neural ranking architectues](rankers.html), including KNRM, Vanilla
   BERT, CEDR, and others
 - Interfaces to several [benchmark datasets](datasets.html), including MS-MARCO,
   TREC Robust 2004, and others
 - Simple [command-line configuration](configuration.html)
 - Integration with [Anserini](https://github.com/castorini/Anserini) for indexing and retrieval.


### Quick Start

**Ready?** Clone and resolve dependencies (python 3.6)

```
$ git clone https://github.com/Georgetown-IR-Lab/OpenNIR.git
$ cd OpenNIR
$ pip install -r requirements.txt
```

**Set?** Let's first try a traditional ranker (`config/trivial`) on the
[antique](https://arxiv.org/abs/1905.08957) QA dataset (`config/antique`). Or try other built-in
datasets, such as TREC CAR (`config/car`), MS MARCO (`config/msmarco`), or TREC Robust 2004
(`config/robust`). Necessary files are downloaded (when avaialble), records are preprocessed, and
initial ranking sets are retrieved --- all automatically.

```
$ bash scripts/pipeline.sh config/antique config/trivial
...
[XXXX-XX-XX XX:XX:XX,XXX][pipeline:default][INFO] valid epoch=-1 judged@10=0.1530 map_rel-3=0.1472 [mrr_rel-3=0.4422] ndcg_gain-1=0:2=1:3=2:4=3@1=0.3688 ndcg_gain-1=0:2=1:3=2:4=3@10=0.2348 ndcg_gain-1=0:2=1:3=2:4=3@3=0.2821 p_rel-3@1=0.3700 p_rel-3@10=0.1310 p_rel-3@3=0.2450
```

**Go!** Train your model. We'll start with [Vanilla BERT](https://arxiv.org/abs/1904.07094). But go
ahead and try [other rankers](rankers.html) too!

```
$ bash scripts/pipeline.sh config/antique config/vanilla_bert
```

**Dig deeper.** See documentation.

 - [Rankers](rankers.html)
 - [Metrics](metrics.html)
 - [Datasets](datasets.html)
 - [Vocabularies](vocab.html)


### Citing OpenNIR

If you use OpenNIR, please cite the following WSDM demonstration paper:

{% raw %}
```
@InProceedings{macavaney:wsdm2020-onir,
  author = {MacAvaney, Sean},
  title = {{OpenNIR}: A Complete Neural Ad-Hoc Ranking Pipeline},
  booktitle = {{WSDM} 2020},
  year = {2020}
}
```
{% endraw %}

## Acknowledgements

I gratefully acknowledge support for this work from the ARCS Endowment Fellowship. I thank Andrew
Yates, Arman Cohan, Luca Soldaini, Nazli Goharian, and Ophir Frieder for valuable feedback on the
manuscript and/or code contributions to OpenNIR.
