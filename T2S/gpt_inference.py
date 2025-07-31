# copied from coqui
import math

import torch
from torch import nn
from transformers import (GemmaForCausalLM, GemmaPreTrainedModel,
                          GPT2PreTrainedModel)
from transformers.modeling_outputs import (CausalLMOutputWithCrossAttentions,
                                           CausalLMOutputWithPast)


class GPT2InferenceModel(GemmaForCausalLM):
    """Override GPT2LMHeadModel to allow for prefix conditioning."""

    def __init__(self, config, gpt, pos_emb, embeddings, norm, linear, kv_cache):
        super().__init__(config)
        self.transformer = gpt
        self.pos_embedding = pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache
        self.cur_pos = None

    def store_prefix_emb(self, prefix_emb):
        self.cached_prefix_emb = prefix_emb

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values.get_seq_length() != 0:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = (
                cache_position[0]
                if cache_position is not None
                else past_key_values.get_seq_length()
            )
            # from IPython import embed; embed()
            max_cache_length = (
                torch.tensor(past_key_values.get_seq_length(), device=input_ids.device)
                if past_key_values.get_seq_length() is not None
                else None
            )
            cache_length = (
                past_length
                if max_cache_length is None
                else torch.min(max_cache_length, past_length)
            )

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = (
            position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        )
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + input_length, device=input_ids.device
            )
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        assert self.cached_prefix_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # assert len(past_key_values) + len(input_ids) == attention_mask.shape[1]

        # Create embedding
        prefix_len = self.cached_prefix_emb.shape[1]
        if input_ids.shape[1] != 1:
            gen_inputs = input_ids[:, prefix_len:]
            gen_emb = self.embeddings(gen_inputs)
            gen_emb = gen_emb  # + self.pos_embedding(gen_emb)
            if self.cached_prefix_emb.shape[0] != gen_emb.shape[0]:
                prefix_emb = self.cached_prefix_emb.repeat_interleave(
                    gen_emb.shape[0] // self.cached_prefix_emb.shape[0], 0
                )
            else:
                prefix_emb = self.cached_prefix_emb.to(gen_emb.dtype)
            emb = torch.cat([prefix_emb, gen_emb], dim=1)
            self.cur_pos = position_ids[:, -1:]
        else:
            emb = self.embeddings(input_ids)
            self.cur_pos += 1.0
            position_ids = self.cur_pos

        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithPast(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
