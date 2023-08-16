# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from fairseq import registry
from fairseq.criterions.fairseq_criterion import FairseqCriterion

CRITERION_REGISTRY = {}
CRITERION_CLASS_NAMES = set()


def build_criterion0(args,task):

    return CRITERION_REGISTRY[args.criterion0].build_criterion0(args, task)
def build_criterion1(args,task):

    return CRITERION_REGISTRY[args.criterion1].build_criterion1(args, task)
# def build_criterion2(args,task):
#
#     return CRITERION_REGISTRY[args.criterion2].build_criterion2(args, task)



def register_criterion(name):
    """Decorator to register a new criterion."""

    def register_criterion_cls(cls):
        if name in CRITERION_REGISTRY:
            raise ValueError('Cannot register duplicate criterion ({})'.format(name))
        if not issubclass(cls, FairseqCriterion):
            raise ValueError('Criterion ({}: {}) must extend FairseqCriterion'.format(name, cls.__name__))
        if cls.__name__ in CRITERION_CLASS_NAMES:
            # We use the criterion class name as a unique identifier in
            # checkpoints, so all criterions must have unique class names.
            raise ValueError('Cannot register criterion with duplicate class name ({})'.format(cls.__name__))
        CRITERION_REGISTRY[name] = cls
        CRITERION_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_criterion_cls


# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('fairseq.criterions.' + module)
# build_criterion0, register_criterion, CRITERION_REGISTRY = registry.setup_registry(
#     '--criterion0',
#     base_class=FairseqCriterion,
#     default='cross_entropy',
# )
# build_criterion1, _ ,_= registry.setup_registry(
#     '--criterion1',
#     base_class=FairseqCriterion,
#     default='cross_entropy',
# )
#
#
# def register_criteron(name):
#     def register_criterion_cls(cls):
#         if name in REGISTRY:
#             raise ValueError('Cannot register duplicate {} ({})'.format(registry_name, name))
#         if cls.__name__ in REGISTRY_CLASS_NAMES:
#             raise ValueError(
#                 'Cannot register {} with duplicate class name ({})'.format(
#                     registry_name, cls.__name__,
#                 )
#             )
#         if base_class is not None and not issubclass(cls, base_class):
#             raise ValueError('{} must extend {}'.format(cls.__name__, base_class.__name__))
#         REGISTRY[name] = cls
#         REGISTRY_CLASS_NAMES.add(cls.__name__)
#         return cls
#
#     return register_criterion_cls
#
# # automatically import any Python files in the criterions/ directory
# for file in os.listdir(os.path.dirname(__file__)):
#     if file.endswith('.py') and not file.startswith('_'):
#         module = file[:file.find('.py')]
#         importlib.import_module('fairseq.criterions.' + module)
