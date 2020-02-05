from tqdm import tqdm
import pandas as pd
import onir


def main():
    logger = onir.log.easy()

    context = onir.injector.load({
        'vocab': onir.vocab,
        'dataset': onir.datasets
    })

    vocab = context['vocab']
    dataset = context['dataset']
    logger.debug(f'vocab: {vocab.config}')
    logger.debug(f'dataset: {dataset.config}')

    dataset.init()

    query_lens = {}
    it = onir.datasets.query_iter(dataset, {'query_len', 'query_id'})
    for record in logger.pbar(it, desc='queries'):
        query_lens[record['query_id']] = float(record['query_len'])
    query_lens = pd.DataFrame([{'query_id': i, 'query_len': l} for i, l in query_lens.items()])
    print(query_lens.describe())

    doc_lens = {}
    # TODO: configure to use doc_iter (all documents) or record_iter (docs from qrels/run)
    it = onir.datasets.doc_iter(dataset, {'doc_len', 'doc_id'})
    # it = onir.datasets.record_iter(dataset, {'doc_len', 'doc_id'}, 'qrels')
    for record in logger.pbar(it, desc='docs'):
        doc_lens[record['doc_id']] = float(record['doc_len'])
    doc_lens = pd.DataFrame([{'doc_id': i, 'doc_len': l} for i, l in doc_lens.items()])
    print(doc_lens.describe())


if __name__ == '__main__':
    main()
