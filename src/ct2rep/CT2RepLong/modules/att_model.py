from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
import tqdm

import ct2rep.CT2RepLong.modules.utils as utils
from ct2rep.CT2Rep.modules.att_model import AttModel as BaseAttModel


class AttModel(BaseAttModel):

    def get_logprobs_state(
            self,
            it,
            fc_feats,
            att_feats,
            memory1,
            seq,
            p_att_feats,
            att_masks,
            memory2=None,
            fc_feats2=None,
            att_feats2=None,
            att_masks2=None,
            state=None,
            output_logsoftmax=1
        ):
        xt = self.embed(it)
        output, state = self.core(xt,
            fc_feats,
            att_feats,
            memory1,
            seq,
            p_att_feats,
            state,
            att_masks2,
            memory2,
            fc_feats2,
            att_feats2,
            att_masks
        )

        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}, targets=None, fc_feats2=None, att_feats2=None):
        beam_size = opt.get('beam_size', 2)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 2)
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        (
            p_fc_feats,
            p_att_feats,
            pp_att_feats,
            p_att_masks,
            memory1,
            _,
            seq_mask1,
            memory2,
            fc_feats2,
            att_feats2,
            att_masks
        ) = self._prepare_feature(fc_feats, att_feats, att_masks, targets, fc_feats2, att_feats2)
        beam_error = 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can ' \
        'be dealt with in future if needed'
        assert beam_size <= self.vocab_size + 1, beam_error
        seq = fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.vocab_size + 1)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]

        state = self.init_hidden(batch_size)
        # print(att_masks.shape,"2")
        # first step, feed bos
        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
        logprobs, state = self.get_logprobs_state(
            it,
            p_fc_feats,
            p_att_feats,
            memory1,
            seq_mask1,
            pp_att_feats,
            p_att_masks,
            memory2,
            fc_feats2,
            att_feats2,
            att_masks,
            state
        )
        # print(memory1.shape,seq_mask.shape,pp_att_feats.shape,p_att_masks.shape,"att")

        (
            p_fc_feats,
            p_att_feats,
            pp_att_feats,
            p_att_masks,
            memory1,
            seq_mask1,
            att_masks,
            memory2,
            fc_feats2,
            att_feats2
        ) = utils.repeat_tensors(beam_size, [
            p_fc_feats,
            p_att_feats,
            pp_att_feats,
            p_att_masks,
            memory1,
            seq_mask1,
            att_masks,
            memory2,
            fc_feats2,
            att_feats2
        ]
    )
        self.done_beams = self.beam_search(
            state,
            logprobs,
            p_fc_feats,
            p_att_feats,
            memory1,
            seq_mask1,
            pp_att_feats,
            p_att_masks,
            memory2,
            fc_feats2,
            att_feats2,
            att_masks,
            opt=opt
        )
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in tqdm.tqdm(range(sample_n)):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs

    def _sample(self, fc_feats, att_feats, context, targets, fc_feats2, att_feats2, att_masks=None):
        opt = self.args.__dict__
        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            # print(fc_feats.shape, att_feats.shape,targets.shape)
            return self._sample_beam(fc_feats, att_feats, att_masks, opt, targets, fc_feats2, att_feats2)
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)

        p_fc_feats, _att_feats, pp_att_feats, p_att_masks, memory1, seq_mask = self._prepare_feature(
            fc_feats, att_feats, targets, att_masks
        )

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(
                sample_n,
                [
                    p_fc_feats, _att_feats,
                    pp_att_feats, p_att_masks
                ]
            )

        trigrams = []  # will be a list of batch_size dictionaries

        seq = fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.vocab_size + 1)
        for t in tqdm.tqdm(range(self.max_seq_length + 1)):
            if t == 0:  # input <bos>
                it = fc_feats.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(
                it,
                p_fc_feats,
                p_att_feats,
                pp_att_feats,
                p_att_masks,
                state,
                memory1,
                seq_mask,
                output_logsoftmax=output_logsoftmax
            )

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:, t - 3:t - 1]
                for i in range(batch_size):  # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t - 1]
                    if t == 3:  # initialize
                        trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:, t - 2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.max_seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx  # This allows eos_idx not being overwritten to 0
                logprobs = logprobs * unfinished.unsqueeze(1).float()
                unfinished = unfinished * (it != self.eos_idx)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _diverse_sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)

        batch_size = fc_feats.size(0)
        self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams_table = [[] for _ in range(group_size)]  # will be a list of batch_size dictionaries

        seq_table = [fc_feats.new_full((batch_size, self.max_seq_length), self.pad_idx, dtype=torch.long) for _ in
                     range(group_size)]
        seqLogprobs_table = [fc_feats.new_zeros(batch_size, self.max_seq_length) for _ in range(group_size)]
        state_table = [self.init_hidden(batch_size) for _ in range(group_size)]

        for tt in tqdm.tqdm(range(self.max_seq_length + group_size)):
            for divm in range(group_size):
                t = tt - divm
                seq = seq_table[divm]
                seqLogprobs = seqLogprobs_table[divm]
                trigrams = trigrams_table[divm]
                if t >= 0 and t <= self.max_seq_length - 1:
                    if t == 0:  # input <bos>
                        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
                    else:
                        it = seq[:, t - 1]  # changed

                    logprobs, state_table[divm] = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats,
                                                                          p_att_masks, state_table[divm])  # changed
                    logprobs = F.log_softmax(logprobs / temperature, dim=-1)

                    # Add diversity
                    if divm > 0:
                        logprobs.clone()
                        for prev_choice in range(divm):
                            prev_decisions = seq_table[prev_choice][:, t]
                            logprobs[:, prev_decisions] = logprobs[:, prev_decisions] - diversity_lambda

                    if decoding_constraint and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                        logprobs = logprobs + tmp

                    # Mess with trigrams
                    if block_trigrams and t >= 3:
                        # Store trigram generated at last step
                        prev_two_batch = seq[:, t - 3:t - 1]
                        for i in range(batch_size):  # = seq.size(0)
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            current = seq[i][t - 1]
                            if t == 3:  # initialize
                                trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                            elif t > 3:
                                if prev_two in trigrams[i]:  # add to list
                                    trigrams[i][prev_two].append(current)
                                else:  # create list
                                    trigrams[i][prev_two] = [current]
                        # Block used trigrams at next step
                        prev_two_batch = seq[:, t - 2:t]
                        mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
                        for i in range(batch_size):
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            if prev_two in trigrams[i]:
                                for j in trigrams[i][prev_two]:
                                    mask[i, j] += 1
                        # Apply mask to log probs
                        # logprobs = logprobs - (mask * 1e9)
                        alpha = 2.0  # = 4
                        logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

                    it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, 1)

                    # stop when all finished
                    if t == 0:
                        unfinished = it != self.eos_idx
                    else:
                        unfinished = seq[:, t - 1] != self.pad_idx & seq[:, t - 1] != self.eos_idx
                        it[~unfinished] = self.pad_idx
                        unfinished = unfinished & (it != self.eos_idx)  # changed
                    seq[:, t] = it
                    seqLogprobs[:, t] = sampleLogprobs.view(-1)

        return torch.stack(seq_table, 1).reshape(batch_size * group_size, -1), torch.stack(
            seqLogprobs_table, 1).reshape(batch_size * group_size, -1)
