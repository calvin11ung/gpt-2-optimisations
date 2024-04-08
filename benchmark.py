from onnxruntime.transformers.models.gpt2.gpt2_helper import GPT2LMHeadModel
from transformers import AutoConfig, AutoTokenizer
from transformers.generation.logits_process import TopPLogitsWarper
import torch
import os
import numpy as np
import json
from ctransformers import AutoModelForCausalLM
import argparse
from datetime import datetime

from calculate_perplexity import compute_perplexity

def get_tokenizer(model_name_or_path, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_ggml_example_inputs(prompt_text, tokenizer, num_attention_heads, hidden_size, num_layer, device):
    encodings_dict = tokenizer._encode_plus(prompt_text)

    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32)

    return input_ids

def get_batched_example_inputs(prompt_text, tokenizer, num_attention_heads, hidden_size, num_layer, batch_size, device):
    
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], dtype=torch.int32)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(position_ids < 0, 0)
    position_ids = position_ids.to(torch.int32)

    # Empty Past State for generating first word
    empty_past = []
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]

    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))

    return input_ids, attention_mask, position_ids, empty_past

def get_example_inputs(prompt_text, tokenizer, num_attention_heads, hidden_size, num_layer, device):
    
    encodings_dict = tokenizer._encode_plus(prompt_text)

    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], dtype=torch.int32)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(position_ids < 0, 0)
    position_ids = position_ids.to(torch.int32)

    # Empty Past State for generating first word
    empty_past = []
    batch_size = 1
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))

    return input_ids, attention_mask, position_ids, empty_past

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
    return sequence


def test_generation_ggml(tokenizer, input_ids, ggml_model, logits_warper, num_tokens_to_produce=30):

    device = 'cpu'

    eos_token_id = tokenizer.eos_token_id

    batch_size = 1

    has_eos = torch.zeros(batch_size, dtype=torch.bool)

    all_token_ids = input_ids.clone()

    for step in range(num_tokens_to_produce):
        with torch.no_grad():

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

    return sequence

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

    model = AutoModelForCausalLM.from_pretrained('ggml/build/models/gpt-2-1558M/ggml-model-q4_0.bin', model_type='gpt2', batch_size=1)
    model.config.max_new_tokens = 30
    return model

def load_text():

    sample_text = 'data/webtext.test.jsonl'

    lines = []

    with open(sample_text, 'r') as f:
        for line in f:
            lines.append(json.loads(line)['text'])

    return lines

    # return sorted(lines, key=len)    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmark', action='store_true', help='Run benchmarking')
    parser.add_argument('-p', '--perplexity', action='store_true', help='Calculate perplexity')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print generated outputs')
    parser.add_argument('-s', '--sort_batches', action='store_true', help='Sort batches')
    parser.add_argument('-n', '--num_batches', type=int, help='Number of batches to test on')
    parser.add_argument('-bsz', '--batch_size', type=int, default=16, help='Number of samples to run batched inference on')

    args = parser.parse_args()

    torch.set_num_threads(10)
    torch.manual_seed(42)

    device = torch.device("cpu")

    # model setup

    model_32_bit = load_32_bit_model()

    model_8_bit = load_8_bit_model()

    # model_4_bit = load_4_bit_model()

    config = model_32_bit.config

    cache_dir = os.path.join(".", "cache_models")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    tokenizer = get_tokenizer("gpt2-xl", cache_dir)

    # data loading setup

    input_text_samples = load_text()

    # tokenise inputs 

    encoded_batches = []
    encoded_sequences = []

    # sorted batches

    cur_batch = []

    for idx, sample in enumerate(input_text_samples):
        input_ids, attention_mask, position_ids, empty_past = get_example_inputs(sample, tokenizer, model_32_bit.config.n_head, model_32_bit.config.n_embd, model_32_bit.config.n_layer, device)

        if input_ids.max() >= model_32_bit.config.vocab_size:
            raise ValueError("One of the input IDs is out of range.")
    
        if len(input_ids) > 968:
            continue

        encoded_sequences.append((input_ids, attention_mask, position_ids, empty_past))

        cur_batch.append(sample)
        if args.sort_batches is False:
            if len(cur_batch) == args.batch_size:
                batch_inputs = get_batched_example_inputs(cur_batch, tokenizer, model_32_bit.config.n_head, model_32_bit.config.n_embd, model_32_bit.config.n_layer, args.batch_size, device)
                encoded_batches.append(batch_inputs)
                cur_batch = []

    if args.sort_batches:

        cur_batch = sorted(cur_batch, key=len)

        # sorted batches
        tmp = []
        for sample in cur_batch:
            tmp.append(sample)
            if len(tmp) == args.batch_size:
                batch_inputs = get_batched_example_inputs(tmp, tokenizer, model_32_bit.config.n_head, model_32_bit.config.n_embd, model_32_bit.config.n_layer, args.batch_size, device)
                encoded_batches.append(batch_inputs)
                tmp = []


    # setup top-p sampling

    top_p_warper = TopPLogitsWarper(top_p=0.9)

    if args.benchmark:

        # Run batched benchmarking on fp32 and uint8 models

        print("Starting batched benchmarks for PyTorch FP32")

        fp_32_batched_start = datetime.now()

        for idx in range(args.num_batches):
            input_ids, attention_mask, position_ids, empty_past = encoded_batches[idx] 

            output = test_generation_pytorch(tokenizer, input_ids, attention_mask, position_ids, empty_past, config, model_32_bit, top_p_warper)

            if args.verbose:
                print(output)
        
        print(f'PyTorch FP32 took {datetime.now() - fp_32_batched_start} to complete {args.num_batches * args.batch_size} samples with batch size {args.batch_size}')

        print("Starting batched benchmarks for PyTorch UINT8")

        uint_8_batched_start = datetime.now()

        for idx in range(args.num_batches):

            input_ids, attention_mask, position_ids, empty_past = encoded_batches[idx] 

            output = test_generation_pytorch(tokenizer, input_ids, attention_mask, position_ids, empty_past, config, model_8_bit, top_p_warper)

            if args.verbose:
                print(output)

        print(f'PyTorch UINT8 took {datetime.now() - uint_8_batched_start} to complete {args.num_batches * args.batch_size} samples with batch size {args.batch_size}')

        # Running single sequence for all models
        # print("Starting single sequence benchmarks for GGML UINT4")

        # uint_4_batched_start = datetime.now()

        # for idx in range(args.num_batches * args.batch_size):

        #     input_ids, attention_mask, position_ids, empty_past = encoded_sequences[idx]
        #     all_token_ids = []
        #     for idx, token in enumerate(model_4_bit.generate(input_ids, top_p=0.9)):
        #         all_token_ids.append(token)
        #         if idx >= 30:
        #             break

        #     sequence = ""

        #     for output in all_token_ids:
        #         sequence += tokenizer.decode(output, skip_special_tokens=True)
            

        #     if args.verbose:
        #         print(output)

        # print(f'GGML UINT4 single sequences took {datetime.now() - uint_4_batched_start} for {args.num_batches * args.batch_size} samples')

        print("Starting single sequence benchmarks for PyTorch FP32")

        uint_32_single_start = datetime.now()

        for idx in range(args.num_batches * args.batch_size):

            input_ids, attention_mask, position_ids, empty_past = encoded_sequences[idx]

            output = test_generation_pytorch(tokenizer, input_ids.unsqueeze(0), attention_mask.unsqueeze(0), position_ids.unsqueeze(0), empty_past, config, model_32_bit, top_p_warper)

            if args.verbose:
                print(output)

        print(f'PyTorch FP32 single sequence took {datetime.now() - uint_32_single_start} for {args.num_batches * args.batch_size} samples')

        print("Starting single sequence benchmarks for PyTorch uint8")

        uint_8_single_start = datetime.now()

        for idx in range(args.num_batches * args.batch_size):

            input_ids, attention_mask, position_ids, empty_past = encoded_sequences[idx]

            output = test_generation_pytorch(tokenizer, input_ids.unsqueeze(0), attention_mask.unsqueeze(0), position_ids.unsqueeze(0), empty_past, config, model_8_bit, top_p_warper)

            if args.verbose:
                print(output)

        print(f'PyTorch UINT8 single sequence took {datetime.now() - uint_8_single_start} for {args.num_batches * args.batch_size} samples')

    if args.perplexity:

        perplexities_32_bit = []
        perplexities_8_bit = []
        # perplexities_4_bit = []

        for idx in range(args.num_batches * args.batch_size):

            perplexities_32_bit.extend(compute_perplexity(model=model_32_bit, tokenizer=tokenizer, add_start_token=False, data=input_text_samples[idx]))
            perplexities_8_bit.extend(compute_perplexity(model=model_8_bit, tokenizer=tokenizer, add_start_token=False, data=input_text_samples[idx]))
            # perplexities_4_bit.extend(compute_perplexity(model=model_4_bit, tokenizer=tokenizer, add_start_token=False, data=input_text_samples[idx]))
            # model_4_bit._context.clear()
            # model_4_bit.ctransformers_llm_reset()
            

        print(f"FP32 Model: Averaged Perplexity: {np.mean(perplexities_32_bit)}, Perplexity Scores: {perplexities_32_bit}")
        print(f"UINT8 Model: Averaged Perplexity: {np.mean(perplexities_8_bit)}, Perplexity Scores: {perplexities_8_bit}")
        # print(f"UINT4 Model: Averaged Perplexity: {np.mean(perplexities_4_bit)}, Perplexity Scores: {perplexities_4_bit}")

if __name__ == '__main__':
    main()