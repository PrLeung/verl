# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib
import os
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["HFRollout"]
LOGITS_LOG_PATH = os.getenv("VERL_LOGITS_LOG_FILE","/vlm/peirouliang/verl/logits_origin.csv")

def _logits_log_csv(step: int, token_6536_logit: float, token_91_logit: float):
    try:
        # If file does not exist or is empty, write header first
        need_header = not os.path.exists(LOGITS_LOG_PATH) or os.path.getsize(LOGITS_LOG_PATH) == 0
        with open(LOGITS_LOG_PATH, "a") as f:
            if need_header:
                # 按用户要求的表头字段
                f.write("step,6576logit,91logit\n")
            f.write(f"{int(step)},{token_6536_logit},{token_91_logit}\n")
    except Exception:
        pass

class FirstTokenMask:
    # 记录修改 token 6536 的 logit 的次数（静态类变量）
    total_rollout_count = 0

    def __init__(self, allowed_ids, batchsize: int = 1, rollout_count: int = 1):
        self.allowed = set(allowed_ids)
        # prompt_lengths: List[int]，batch内每条样本的原始prompt长度
        self.mask = None  # 用于标记哪些token是允许的
        self.batchsize = int(batchsize) if batchsize is not None else 1
        self.rollout_count = int(rollout_count) if rollout_count is not None else 1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # input_ids: [bsz, cur_len], scores: [bsz, vocab]
        # bsz, vocab = scores.shape
        bsz = 1
        # assert 1==2, f'input_ids:{input_ids}, scores:{scores.shape}, self.prompt_lengths:{self.prompt_lengths}'
        # 用dtype对应的最小值，避免半精度下的 -inf 数值问题
        neg_inf = torch.finfo(scores.dtype).min
        for i in range(bsz):
            # 当前样本是否正处于第一个生成步
            # 对decoder-only，一般满足 cur_len == prompt_len + 1（有的实现内步长定义略有不同，可兼容两种）
            cur_len = len(input_ids)
            if cur_len == 0:
                if self.mask is None:
                    self.mask = torch.tensor([
                        False if i in self.allowed else True for i in range(scores.shape[-1])
                    ], dtype=torch.bool, device=scores.device)
                
                FirstTokenMask.total_rollout_count += 1
                # dist.all_reduce(FirstTokenMask.total_rollout_count, op=dist.ReduceOp.SUM)

                # 计算当前步数 = floor(总生成次数 / (batchsize * rollout数量))
                world_size = int(os.environ.get("WORLD_SIZE", "8"))
                denom = max(1, self.batchsize * self.rollout_count / world_size)
                current_step = FirstTokenMask.total_rollout_count // denom
                token_6536_logit = scores[6536].item()
                token_91_logit = scores[91].item()
                _logits_log_csv(current_step, token_6536_logit, token_91_logit)
                
                scores[self.mask] = neg_inf
                
        return scores


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module
        cfg = self.config
        self.answer_suffix_mode = cfg.get("answer_suffix_mode", "stage1")
        batchsize = (
            cfg.get("batch_size", None)
            or 1
        )
        print(f"batchsize: {batchsize}")
        rollout_count = cfg.get("n", None) or 1
        print(f"rollout_count: {rollout_count}")
        self.logits_processor = FirstTokenMask(
            allowed_ids=[6536,91],
            batchsize=int(batchsize),
            rollout_count=int(rollout_count),
        ) if self.answer_suffix_mode == "stage4" else None

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # make sampling args can be overridden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

        if not do_sample:
            # do_sample==False -> greedy decoding
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # do validate and do sample -> use val_kwargs
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # do_sample -> use rollout config
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": self.config.n,
            }

        # make config according to generate mode
        generation_config = GenerationConfig(**kwargs)

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = self.module.generate(
                input_ids=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                do_sample=do_sample,
                max_new_tokens=response_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=generation_config,
                output_scores=False,  # this is potentially very large
                return_dict_in_generate=True,
                use_cache=True,
                logits_processor=[self.logits_processor] if self.logits_processor is not None else None,
            )

        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        # make necessary reputations if num_return_sequences > 1
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()

        self.module.train()
        return DataProto(batch=batch)
