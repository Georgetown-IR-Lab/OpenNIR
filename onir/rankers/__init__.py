# pylint: disable=C0413
from onir import util
registry = util.Registry(default='trivial')
register = registry.register

from onir.rankers.base import Ranker
from onir.rankers import base, conv_knrm, drmm, duetl, knrm, matchpyramid, pacrr, trivial
from onir.rankers import vanilla_transformer, cedr_drmm, cedr_knrm, cedr_pacrr, epic
from onir.rankers import matchzoo
from onir.rankers import hgf4_joint
