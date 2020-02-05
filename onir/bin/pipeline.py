import onir


def main():
    context = onir.injector.load({
        'vocab': onir.vocab,
        'train_ds': onir.datasets,
        'ranker': onir.rankers,
        'trainer': onir.trainers,
        'valid_ds': onir.datasets,
        'valid_pred': onir.predictors,
        'test_ds': onir.datasets,
        'test_pred': onir.predictors,
        'pipeline': onir.pipelines,
    }, pretty=True)

    context['pipeline'].run()


if __name__ == '__main__':
    main()
