import onir


def main():
    logger = onir.log.easy()

    context = onir.injector.load({
        'vocab': onir.vocab,
        'dataset': onir.datasets,
    })

    v = vocab = context['vocab']
    d = ds = dataset = context['dataset']

    logger.debug(f'vocab: {vocab.config}')
    logger.debug(f'dataset: {dataset.config}')

    import pdb
    pdb.set_trace()
    pass


if __name__ == '__main__':
    main()
