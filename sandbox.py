from onnxruntime.transformers.models.gpt2.gpt2_helper import Gpt2Helper, GPT2LMHeadModel
from transformers import AutoConfig, AutoTokenizer
from transformers.generation.logits_process import TopPLogitsWarper
import torch
import os
from datetime import datetime, timedelta
import onnxruntime
import numpy
from onnxruntime.transformers.io_binding_helper import TypeHelper
from onnxruntime.transformers.io_binding_helper import IOBindingHelper
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from evaluate import logging
import json
from ctransformers import AutoModelForCausalLM

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

def inference_with_io_binding(session, config, input_ids, position_ids, attention_mask, past):
    output_shapes = Gpt2Helper.get_output_shapes(
        batch_size=input_ids.size(0),
        past_sequence_length=past[0].size(3),
        sequence_length=input_ids.size(1),
        config=config,
    )
    output_buffers = Gpt2Helper.get_output_buffers(output_shapes, 'cpu')

    io_binding = IOBindingHelper.prepare_io_binding(
        session, input_ids, position_ids, attention_mask, past, output_buffers, output_shapes
    )
    session.run_with_iobinding(io_binding)

    outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(session, output_buffers, output_shapes, return_numpy=False)
    return outputs

def greedy(next_token_logits):

    next_tokens = torch.argmax(next_token_logits, dim=-1)

    return next_tokens

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

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

    print("Text generation using PyTorch")
    eos_token_id = tokenizer.eos_token_id

    batch_size = input_ids.size(0)

    print(batch_size)

    has_eos = torch.zeros(batch_size, dtype=torch.bool)

    all_token_ids = input_ids.clone()

    ppl = []

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

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

        # if torch.all(has_eos):
        #     break

    sequence = ""

    for i, output in enumerate(all_token_ids):
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
        # if torch.all(has_eos):
        #     break

    sequence = ""

    for i, output in enumerate(all_token_ids):
        sequence += tokenizer.decode(output, skip_special_tokens=True)

    print(sequence)


def main():
    torch.set_num_threads(10)
    torch.manual_seed(42)
    device = torch.device("cpu")

    sample_text = '/home/calvin/gpt2/data/gpt-2-output-dataset/data/webtext.test.jsonl'

    input_text_samples = []
    bucket = []
    batch_size = 1

    cache_dir = os.path.join(".", "cache_models")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    model_name_or_path = "gpt2-xl"
    # model_name_or_path = "gpt2-quantized-2.pt"
    quant_model = torch.load("gpt2-xl-quantized.pt")
    quant_model.eval().to(device)
    
    config = AutoConfig.from_pretrained("gpt2-xl", cache_dir=cache_dir)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

    num_attention_heads = model.config.n_head
    hidden_size = model.config.n_embd
    num_layer = model.config.n_layer

    tokenizer = get_tokenizer("gpt2-xl", cache_dir)

    # First, read all lines and sort them by length
    lines = []

    with open(sample_text, 'r') as f:
        for line in f:
            lines.append(json.loads(line)['text'])

    sorted_lines = sorted(lines, key=len)

    ggml_text_samples = []

    # Now, group the sorted lines into sublists of 20
    for text in sorted_lines:
        bucket.append(text)
        if len(bucket) == batch_size:
            input_text_samples.append(bucket)
            bucket = []
        ggml_text_samples.append(get_ggml_example_inputs(text, tokenizer, num_attention_heads, hidden_size, num_layer, device))

    # Check for any remaining items in bucket not yet added to input_text_samples
    if bucket:
        input_text_samples.append(bucket)

    computed_batches = []

    for batch in input_text_samples:

        computed_batches.append(get_example_inputs(batch, tokenizer, num_attention_heads, hidden_size, num_layer, device))

    ggml_model = AutoModelForCausalLM.from_pretrained('/home/calvin/gpt2/ggml/build/models/gpt-2-1558M/ggml-model-q4_0.bin', model_type='gpt2')
    ggml_model.config.max_new_tokens = 30
    top_p_warper = TopPLogitsWarper(top_p=0.9)


    input_ids, attention_mask, position_ids, empty_past = computed_batches[0]

    start = datetime.now()
    output = test_generation_pytorch(tokenizer, input_ids, attention_mask, position_ids, empty_past, config, quant_model, top_p_warper)
    print(f'pytorch uint8 took {datetime.now() - start}')

    start = datetime.now()
    output = test_generation_pytorch(tokenizer, input_ids, attention_mask, position_ids, empty_past, config, model, top_p_warper)
    print(f'pytorch fp32 took {datetime.now() - start}')

    start = datetime.now()

    # test_generation_ggml(tokenizer, ggml_text_samples[0], ggml_model, top_p_warper)
    all_token_ids = list(ggml_text_samples[0])
    for idx, token in enumerate(ggml_model.generate(ggml_text_samples[0], top_p=0.9)):
        all_token_ids.append(token)
        if idx >= 30:
            break

    sequence = ""

    for i, output in enumerate(all_token_ids):
        sequence += tokenizer.decode(output, skip_special_tokens=True)

    print(sequence)
    print(f'ggml took {datetime.now() - start}')

    exit()

    # breakpoint()
    # for idx, input_ids in enumerate(ggml_text_samples):
    #     start = datetime.now()

    #     test_generation_ggml(tokenizer, input_ids, ggml_model, top_p_warper)
    #     print(f'took {datetime.now() - start}')

    # total_time = timedelta()
    # perplexity = []

    # for idx, (input_ids, attention_mask, position_ids, empty_past) in enumerate(computed_batches):

    #     start = datetime.now()
    #     test_generation_pytorch(tokenizer, input_ids, attention_mask, position_ids, empty_past, config, model, top_p_warper)
    #     total_time += (datetime.now() - start)

    #     perplexity.extend(compute_perplexity(model=model, tokenizer=tokenizer, add_start_token=False, data=input_text_samples[idx]))

    #     if idx == 0:
    #         break

    # print(f'pytorch FP32 took {total_time}')
    # print(f'mean perplexity: {numpy.mean(perplexity)}')

    total_time = timedelta()
    perplexity = []

    for idx, (input_ids, attention_mask, position_ids, empty_past) in enumerate(computed_batches):

        start = datetime.now()
        test_generation_pytorch(tokenizer, input_ids, attention_mask, position_ids, empty_past, config, quant_model, top_p_warper)
        total_time += (datetime.now() - start)

        perplexity.extend(compute_perplexity(model=quant_model, tokenizer=tokenizer, add_start_token=False, data=input_text_samples[idx]))

        if idx == 0:
            break

    print(f'pytorch UINT8 took {total_time}')
    print(f'mean perplexity: {numpy.mean(perplexity)}')



    # start = datetime.now()
    # test_generation_pytorch(tokenizer, input_ids, attention_mask, position_ids, empty_past, config, quant_model, top_p_warper)
    # print(f'pytorch uint8 took {datetime.now() - start}')
    # results = compute_perplexity(model=quant_model, tokenizer=tokenizer, add_start_token=False, data=input_text_samples)
    # print(f"UINT8 perplexity: {results}")
        


if __name__ == '__main__':
    main()