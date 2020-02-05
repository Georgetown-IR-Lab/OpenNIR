import sys
from onir import rankers
from onir.interfaces import matchzoo

if matchzoo.is_available:
    rankers.register('mz_knrm')(matchzoo.generate_mz_ranker(matchzoo.mz.models.KNRM))
    rankers.register('mz_conv_knrm')(matchzoo.generate_mz_ranker(matchzoo.mz.models.ConvKNRM))
    # This one doesn't seem to train properly...
    # rankers.register('mz_drmmtks')(generate_mz_ranker(matchzoo.mz.models.DRMMTKS))
else:
    class FaxuMzRanker:
        """
        MatchZoo-py not installed, so no MatchZoo rankers included.
        Install with: pip install matchzoo-py==1.0
        """
        def __init__(self, logger):
            logger.error('MatchZoo-py not installed, so no MatchZoo rankers included. '
                         'Install with: pip install matchzoo-py==1.0')
            sys.exit(1)

        def default_config():
            return {}

    rankers.register('mz_knrm')(FaxuMzRanker)
    rankers.register('mz_conv_knrm')(FaxuMzRanker)
