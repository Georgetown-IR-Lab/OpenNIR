import onir


def main():
    context = onir.injector.load({
        'vocab': onir.vocab,
        'ranker': onir.rankers,
        'train_ds': onir.datasets,
        'trainer': onir.trainers,
        'valid_ds': onir.datasets,
        'valid_pred': onir.predictors,
        'test_ds': onir.datasets,
        'test_pred': onir.predictors,
        'pipeline': onir.pipelines.ExtractBertWeights,
    })

    logger = onir.log.easy()

    logger.debug(f'vocab: {context["vocab"].config}')
    logger.debug(f'ranker: {context["ranker"].config}')
    logger.debug(f'train_ds: {context["train_ds"].config}')
    logger.debug(f'trainer: {context["trainer"].config}')
    logger.debug(f'valid_ds: {context["valid_ds"].config}')
    logger.debug(f'valid_pred: {context["valid_pred"].config}')
    logger.debug(f'test_ds: {context["test_ds"].config}')
    logger.debug(f'test_pred: {context["test_pred"].config}')
    logger.debug(f'pipeline: {context["pipeline"].config}')

    context['pipeline'].run()


if __name__ == '__main__':
    main()
