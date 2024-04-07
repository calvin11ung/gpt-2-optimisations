from onnxruntime.transformers.models.gpt2.gpt2_helper import Gpt2Helper, GPT2LMHeadModel
from transformers import AutoConfig, AutoTokenizer
from transformers.generation.logits_process import TopPLogitsWarper
import torch
import os
from datetime import datetime, timedelta
import numpy as np
from onnxruntime.transformers.io_binding_helper import TypeHelper
from onnxruntime.transformers.io_binding_helper import IOBindingHelper
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from evaluate import logging
import json
from ctransformers import AutoModelForCausalLM
import argparse

def get_tokenizer(model_name_or_path, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_ggml_example_inputs(prompt_text, tokenizer, num_attention_heads, hidden_size, num_layer, device):
    encodings_dict = tokenizer._encode_plus(prompt_text)

    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32)

    return input_ids

def get_example_inputs(prompt_text, tokenizer, num_attention_heads, hidden_size, num_layer, device):
    
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], dtype=torch.int32)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(position_ids < 0, 0)
    position_ids = position_ids.to(torch.int32)

    # Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))

    return input_ids, attention_mask, position_ids, empty_past


def compute_perplexity(
        data, model, tokenizer, batch_size: int = 16, add_start_token: bool = True, device=None, max_length=None
    ):

        device = 'cpu'

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            data,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return ppls

def test_generation_pytorch(tokenizer, input_ids, attention_mask, position_ids, past, config, torch_model, logits_warper, num_tokens_to_produce=30):

    device = 'cpu'

    eos_token_id = tokenizer.eos_token_id

    batch_size = input_ids.size(0)

    has_eos = torch.zeros(batch_size, dtype=torch.bool)

    all_token_ids = input_ids.clone()


    for step in range(num_tokens_to_produce):
        with torch.no_grad():
            outputs = torch_model(
                input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past
            )

        next_token_logits = outputs[0][:, -1, :]

        warped_logits = logits_warper(input_ids, next_token_logits)
        
        probabilities = torch.softmax(warped_logits, dim=-1)
        next_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(1)

        has_eos = has_eos | (next_tokens == eos_token_id)
        tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
        all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(device)
        position_ids = (position_ids[:, -1] + 1).reshape(batch_size, 1)
        attention_mask = torch.cat([attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1).to(device)

        past = list(outputs[1])

        if torch.all(has_eos):
            break

    sequence = ""

    for output in all_token_ids:
        sequence += tokenizer.decode(output, skip_special_tokens=True)
    print(sequence)


def test_generation_ggml(tokenizer, input_ids, ggml_model, logits_warper, num_tokens_to_produce=30):

    device = 'cpu'

    print("Text generation using ggml")
    eos_token_id = tokenizer.eos_token_id

    batch_size = 1

    has_eos = torch.zeros(batch_size, dtype=torch.bool)

    all_token_ids = input_ids.clone()

    for step in range(num_tokens_to_produce):
        with torch.no_grad():
            start = datetime.now()
            ggml_model.eval(
                tokens=input_ids, batch_size=1
            )

        next_token_logits = torch.Tensor(ggml_model.logits).unsqueeze(0)

        warped_logits = logits_warper(input_ids, next_token_logits)
        probabilities = torch.softmax(warped_logits, dim=-1)
        next_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(1)

        has_eos = has_eos | (next_tokens == eos_token_id)
        tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
        all_token_ids = torch.cat([all_token_ids, tokens_to_add], dim=-1)

        input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(device)
        if torch.all(has_eos):
            break

    sequence = ""

    for output in all_token_ids:
        sequence += tokenizer.decode(output, skip_special_tokens=True)

    print(sequence)

def load_32_bit_model():

    model_name_or_path = "gpt2-xl"

    cache_dir = os.path.join(".", "cache_models")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    config = AutoConfig.from_pretrained("gpt2-xl", cache_dir=cache_dir)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

    return model

def load_8_bit_model():

    model = torch.load("gpt2-xl-quantized.pt")
    model.eval().to('cpu')

    return model

def load_4_bit_model():

    model = AutoModelForCausalLM.from_pretrained('/home/calvin/gpt2/ggml/build/models/gpt-2-1558M/ggml-model-q4_0.bin', model_type='gpt2')

    return model

def load_text(batch_size):

    sample_text = '~/gpt2-optimisation/data/gpt-2-output-dataset/data/webtext.test.jsonl'

    lines = []

    with open(sample_text, 'r') as f:
        for line in f:
            lines.append(json.loads(line)['text'])

    return sorted(lines, key=len)    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmark', action='store_true', help='Run benchmarking')
    parser.add_argument('-p', '--perplexity', action='store_true', help='Calculate perplexity')
    parser.add_argument('-n', '--num_batches', type=int, help='Number of batches to test on')
    parser.add_argument('-bsz', '--batch_size', type=int, default=16, help='Number of samples to run batched inference on')

    args = parser.parse_args()

    torch.set_num_threads(10)
    torch.manual_seed(42)

    device = torch.device("cpu")

    # model setup

    model_32_bit = load_32_bit_model()

    model_8_bit = load_8_bit_model()

    model_4_bit = load_4_bit_model()

    config = model_32_bit.config

    cache_dir = os.path.join(".", "cache_models")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    tokenizer = get_tokenizer("gpt2-xl", cache_dir)

    # data loading setup

    input_text_samples = load_text(args.batch_size)

    # tokenise inputs 

    encoded_batches = []
    encoded_sequences = []

    cur_batch = []

    for idx, sample in enumerate(input_text_samples):
        
        encoded_sequences.append(get_example_inputs(sample, tokenizer, model_32_bit.config.n_head, model_32_bit.config.n_embd, model_32_bit.config.n_layer, device))

        cur_batch.append(sample)

        if idx == args.batch_size:

            encoded_batches.append(get_example_inputs(cur_batch, tokenizer, model_32_bit.config.n_head, model_32_bit.config.n_embd, model_32_bit.config.n_layer, device))

            cur_batch = []

    # setup top-p sampling

    top_p_warper = TopPLogitsWarper(top_p=0.9)

    if args.benchmark:

        # Run batched benchmarking on fp32 and uint8 models

        print("Starting batched benchmarks for PyTorch FP32")

        fp_32_batched_start = datetime.now()

        for idx in range(args.num_samples):

            input_ids, attention_mask, position_ids, empty_past = encoded_batches[idx] 

            test_generation_pytorch(tokenizer, input_ids, attention_mask, position_ids, empty_past, config, model_32_bit, top_p_warper)
        
        print(f'PyTorch FP32 took {datetime.now() - fp_32_batched_start} to complete {args.num_samples * args.batch_size} samples with batch size {args.batch_size}')

        print("Starting batched benchmarks for PyTorch UINT8")

        uint_8_batched_start = datetime.now()

        for idx in range(args.num_samples):

            input_ids, attention_mask, position_ids, empty_past = encoded_batches[idx] 

            test_generation_pytorch(tokenizer, input_ids, attention_mask, position_ids, empty_past, config, model_8_bit, top_p_warper)
        
        print(f'PyTorch UINT8 took {datetime.now() - uint_8_batched_start} to complete {args.num_samples * args.batch_size} samples with batch size {args.batch_size}')

        # Running single sequence for all models
        print("Starting single sequence benchmarks for GGML UINT4")

        uint_4_batched_start = datetime.now()

        for idx in range(args.num_samples * args.batch_size):

            input_ids, attention_mask, position_ids, empty_past = encoded_sequences[idx]

            test_generation_ggml(tokenizer, input_ids, model_4_bit, top_p_warper)

        print(f'GGML UINT4 single sequences took {datetime.now() - uint_4_batched_start} for {args.num_samples * args.batch_size} samples')

        print("Starting single sequence benchmarks for PyTorch FP32")

        uint_32_single_start = datetime.now()

        for idx in range(args.num_samples * args.batch_size):

            input_ids, attention_mask, position_ids, empty_past = encoded_sequences[idx]

            test_generation_pytorch(tokenizer, input_ids, attention_mask, position_ids, empty_past, config, model_32_bit, top_p_warper)

        print(f'PyTorch FP32 single sequence took {datetime.now() - uint_32_single_start} for {args.num_samples * args.batch_size} samples')

        print("Starting single sequence benchmarks for PyTorch uint8")

        uint_8_single_start = datetime.now()

        for idx in range(args.num_samples * args.batch_size):

            input_ids, attention_mask, position_ids, empty_past = encoded_sequences[idx]

            test_generation_pytorch(tokenizer, input_ids, attention_mask, position_ids, empty_past, config, model_8_bit, top_p_warper)

        print(f'PyTorch FP32 single sequence took {datetime.now() - uint_8_single_start} for {args.num_samples * args.batch_size} samples')

    if args.perplexity:

        perplexities_32_bit = []
        perplexities_8_bit = []
        perplexities_4_bit = []

        for idx in range(args.num_samples * args.batch_size):

            perplexities_32_bit.extend(compute_perplexity(model=model_32_bit, tokenizer=tokenizer, add_start_token=False, data=input_text_samples[idx]))
            perplexities_8_bit.extend(compute_perplexity(model=model_8_bit, tokenizer=tokenizer, add_start_token=False, data=input_text_samples[idx]))
            perplexities_4_bit.extend(compute_perplexity(model=model_4_bit, tokenizer=tokenizer, add_start_token=False, data=input_text_samples[idx]))

        print(f"FP32 Model: Averaged Perplexity: {np.mean(perplexities_32_bit)}, Perplexity Scores: {perplexities_32_bit}")
        print(f"UINT8 Model: Averaged Perplexity: {np.mean(perplexities_8_bit)}, Perplexity Scores: {perplexities_8_bit}")
        print(f"UINT4 Model: Averaged Perplexity: {np.mean(perplexities_4_bit)}, Perplexity Scores: {perplexities_4_bit}")