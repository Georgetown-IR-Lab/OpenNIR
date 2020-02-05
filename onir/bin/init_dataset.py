import onir


def main():
    logger = onir.log.easy()

    context = onir.injector.load({
        'vocab': onir.vocab,
        'dataset': onir.datasets,
    }, pretty=True)

    context['dataset'].init()


if __name__ == '__main__':
    main()
