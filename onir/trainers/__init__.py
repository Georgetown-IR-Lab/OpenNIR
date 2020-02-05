# pylint: disable=C0413
from onir import util
registry = util.Registry(default='pairwise')
register = registry.register

from onir.trainers import misc
from onir.trainers.base import Trainer
from onir.trainers.trivial import TrivialTrainer
from onir.trainers.pointwise import PointwiseTrainer
from onir.trainers.pairwise import PairwiseTrainer
