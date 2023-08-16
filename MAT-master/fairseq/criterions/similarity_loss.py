# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
from sentence_transformers import SentenceTransformer
import math

from fairseq import utils

from . import FairseqCriterion, register_criterion
from torch import nn
import torch

import torch.nn.functional as F
class PearsonCorrelation(nn.Module):
    def forward(self,tensor_1,tensor_2):
        x = tensor_1
        y = tensor_2

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost

@register_criterion('similarity_loss')
class SimilarityLoss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)


    def forward(self,sentence_vec,eng_sen):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # net_output = model(**sample['net_input'])
        loss = self.compute_loss( sentence_vec,eng_sen)
        return loss

    def compute_loss(self,  sentence_vec,eng_sen):

        sentence_vec=torch.tensor(sentence_vec).cuda()
        # eng_sen=torch.tensor(eng_sen.tolist()).cuda()

        # B=net_output[0].size(1)
        # T = net_output[0].size(2)
        # C = net_output[0].size(3)
        # state=torch.zeros([B,T,C]).cuda()
        # for i in net_output[0]:
        #     state=state+i
        # state=state/net_output[0].size(0)
        # word_sen_vec=self.transform(state).cuda()
        # connected_layer=nn.Linear(in_features=word_sen_vec.size(1), out_features=512).cuda()
        # word_sen_vec=connected_layer(word_sen_vec.float()).cuda()  #192 x 512

        # logp_x = F.log_softmax(eng_sen, dim=-1)
        #         # p_y = F.softmax(sentence_vec, dim=-1)
        #         # # similarity_loss=torch.cosine_similarity(eng_sen,sentence_vec) #160239 x 512
        #         # # loss=1-((similarity_loss+1)/2)
        #         #
        #         # kl_mean = F.kl_div(logp_x, p_y, reduction='mean')
        # loss=torch.sum(loss)
        # pearson = PearsonCorrelation()
        # PC = pearson(sentence_vec, eng_sen)
        # PC=1-abs(PC)
        similarity_loss = torch.cosine_similarity(eng_sen, sentence_vec).cuda()  # 160239 x 512
        loss = 1 - ((similarity_loss + 1) / 2).cuda()
        loss = torch.sum(loss).cuda()
        return loss



    def transform(self,decoder_out):
       # encoder_out=decoder_out.transpose(0,1) #B*T*C
        jieguo=[]

        B = decoder_out.size(0)  # 192
        T=decoder_out.size(1) #16
        C = decoder_out.size(2) #10152
        for i in decoder_out:
            for j in i:
                sum=torch.sum(j)
                sum = sum / C
                jieguo.append(sum)
        jieguo = torch.tensor(jieguo).cuda()
        jieguo = jieguo.reshape(B,T).cuda()
        return jieguo

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
