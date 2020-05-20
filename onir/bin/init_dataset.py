import onir


def main():
    logger = onir.log.easy()

    context = onir.injector.load({
        'vocab': onir.vocab,
        'dataset': onir.datasets,
    }, pretty=True)

    context['dataset'].init()

    num_docs = context['dataset'].num_docs()
    logger.info(f'dataset is initialized ({num_docs} documents)')


if __name__ == '__main__':
    main()
