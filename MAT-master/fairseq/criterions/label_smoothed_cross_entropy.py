# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from fairseq import utils

from . import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample,src_dict,tgt_dict,reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss,eng_sen = self.compute_loss(model, net_output, sample, src_dict,tgt_dict,reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output,eng_sen

    # def compute_loss(self, model, net_output, sample, reduce=True):
    #     lprobs = model.get_normalized_probs(net_output, log_probs=True)
    #     lprobs = lprobs.view(-1, lprobs.size(-1))
    #     target = model.get_targets(sample, net_output).view(-1, 1)
    #     loss, nll_loss = label_smoothed_nll_loss(
    #         lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
    #     )
    #     return loss, nll_loss
    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        # if self.ignore_prefix_size > 0:
        #     if getattr(lprobs, "batch_first", False):
        #         lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
        #         target = target[:, self.ignore_prefix_size :].contiguous()
        #     else:
        #         lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
        #         target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1),lprobs,target

    def compute_loss(self, model, net_output, sample,src_dict,tgt_dict, reduce=True):
        lprobs, target ,lprobs_no,target_no= self.get_lprobs_and_target(model, net_output, sample)
        sentences=[]
        for sen in lprobs_no:
            sentence=[]
            for word in sen:
                a=word.argmax(dim=0)
                sentence.append(a.item())
            sentence1=torch.tensor(sentence)
            sentences.append(sentence1)
        eng_sen=[]
        for i in range(len(sample['id'].tolist())):
            src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
            src_str = src_dict.string(src_tokens, bpe_symbol='@@ ')
            _, sen_str, _ = utils.post_process_prediction(
                    hypo_tokens=sentences[i].int().cpu(),
                    src_str=src_str,
                    alignment=None,
                    align_dict=None,
                    tgt_dict=tgt_dict,
                    remove_bpe='@@ ',
            )
            eng_sen.append(sen_str)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss,eng_sen

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
