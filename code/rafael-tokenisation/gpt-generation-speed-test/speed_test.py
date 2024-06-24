import gc
import os
import textwrap
import time
import json

import torch
from memory_profiler import memory_usage
from transformers import GPTNeoForCausalLM, RobertaForMaskedLM, RobertaTokenizerFast, AlbertTokenizerFast, RobertaForCausalLM, AutoModelForCausalLM, AutoTokenizer
import pdb

is_sentencepiece = True

def chars_per_text(text, encoding='utf-8'):
    # Calculate the number of bits for the token based on the encoding
    return len(text)

def clean_text(text):
    return text.replace("Ġ", "").replace("▁", "").replace("#", "").replace(" ", "")

def remove_hash(text):
    return text.replace("#", "")

def calculate_chars(texts):
    chars_generated = 0
    for text in texts:
        cleaned_text = clean_text(text)
        text_chars = chars_per_text(cleaned_text)
        chars_generated += text_chars
    return chars_generated

def generate_batch(model, tok, name, prompts, min_mem_usage):
    # print(len(prompts))
    print(f'Generating the batch for the model {name}')
    gc.collect()  # Explicit garbage collection
    torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
    # Tokenize the prompts
    input_ids_list = [tok.encode(prompt, return_tensors='pt') for prompt in prompts]

    if is_sentencepiece:
        for t in input_ids_list:
            t[0, 0] = 2
            t[0, -1] = 3

    # Determine the maximum sequence length
    max_length = max(len(ids[0]) for ids in input_ids_list)
    input_target_length = 32

    # Ensure that in case for some magical reason max_length is greater than the selected input target
    # we still use the maximum.
    max_length = max(max_length, input_target_length)
    # print("Input len: ", max_length)
    padded_input_ids = [torch.nn.functional.pad(ids, (0, max_length - len(ids[0])), 'constant', tok.pad_token_id) for
                        ids in input_ids_list]
    input_ids = torch.stack(padded_input_ids, dim=0).squeeze(1)
    # Stack the padded sequences to create a batch
    input_ids = torch.stack(padded_input_ids, dim=0).squeeze(1)

    # Move input_ids to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = input_ids.to(device)
    model.to(device)

    # Generate outputs in batch
    batch_size = input_ids.size(0) # THE BATCH SIZE IS 256
    # print(batch_size) # 
    fixed_output_length = 256
    try:
        with torch.no_grad():
            outputs = []

            def generate():
                output = model.generate(
                    input_ids,
                    min_length=fixed_output_length,  # Set the minimum length
                    max_length=fixed_output_length,  # Set the maximum length as the same to insure fixed output length
                    eos_token_id=tok.eos_token_id, 
                    temperature=1,
                    do_sample=True  # Enable sampling to allow for variability
                )
                outputs.append(output)
                return output

            start_time = time.time()
            memory_usage_info = memory_usage((generate,), interval=0.01, timeout=None)
            end_time = time.time()
            if min_mem_usage[0] is None:
                min_mem_usage[0] = min(memory_usage_info)
            else:
                min_mem_usage[0] = min(min_mem_usage[0], min(memory_usage_info))

            # Calculate memory usage and time taken
            peak_memory_usage = max(memory_usage_info) - min_mem_usage[0]
            time_taken = end_time - start_time

        outputs = outputs[0]

        # Decode and display the outputs
        output_texts = []
        output_lengths = []
        avg_number_tokens = 0

        for output in outputs:
            decoded_output = tok.decode(output)
            
            number_tokens = len(tok.encode(remove_hash(decoded_output)))
            avg_number_tokens += number_tokens
            # pdb.set_trace()

            output_texts.append(decoded_output)
            output_lengths.append(len(output))

        # pdb.set_trace()
        avg_number_tokens /= len(outputs)
        # for i, output_text in enumerate(output_texts):
        # print(f"Prompt {i+1}: {prompts[i]}")
        # print(f"Generated Text {i+1}:\n {output_text}")
        # print(f"Number of Tokens {i+1}: {output_lengths[i]}")
        # print("\n" + "-"*50 + "\n")
        # print("Number of outputs: ", len(output_lengths))

        # Return time taken per one token. 
        # Divide by output_lengths[0] because all outputs in a batch have the same length.
        # Divide by the number of outputs in a batch.
        chars_generated = calculate_chars(output_texts)
        return peak_memory_usage, (avg_number_tokens * len(outputs)) / time_taken , chars_generated / time_taken, output_texts

    except RuntimeError as e:
        print(f"Batch size {batch_size} failed due to: {e}")

def run_batches(model, tok, name, prompt_batches, tokens_per_s_acc, chars_per_s_acc):
    warmup_prompts = [
        'Once upon a time, there was a guy named',
        'In a distant land, there was a mysterious forest',
        'The scientists in a school made',
        # Add more prompts as needed
    ]
    # Warm-up runs
    print("Warm-up runs:")
    for _ in range(3):
        generate_batch(model=model, tok=tok, name=name, prompts=warmup_prompts * 50, min_mem_usage=[None])
        print("Still warming up...")

    print("-----------------------------------------------------------------")

    min_mem_usage = [None]
    for prompt_batch in prompt_batches:
        print(len(prompt_batches))
        print(prompt_batch)
        memory1, tokens_per_s, chars_per_s, outputs = generate_batch(model, tok, name, prompt_batch, min_mem_usage=min_mem_usage) 
        tokens_per_s_acc.append(tokens_per_s)
        chars_per_s_acc.append(chars_per_s)
        print("Memory received: ", memory1)
        print("Tokens per second: ", tokens_per_s_acc)
        print("Chars per second: ", chars_per_s_acc)       


def compare_models(models, toks, names, prompt_batches):
    tokens_per_s_accs = [[] for _ in models]
    chars_per_s_accs = [[] for _ in models]
    for i, model in enumerate(models):
        run_batches(model, toks[i], names[i], prompt_batches, tokens_per_s_accs[i], chars_per_s_accs[i])

    average_tokens_per_s = [sum(t_s) / len(t_s) for t_s in tokens_per_s_accs]
    average_chars_per_s = [sum(b_s) / len(b_s) for b_s in chars_per_s_accs]

    for i, model in enumerate(models):
        print(f"Average inference speed for {names[i]} is: {average_tokens_per_s[i]} seconds per token and {average_chars_per_s[i]} chars per second")

    return average_tokens_per_s, average_chars_per_s

if __name__ == "__main__":

    model_dir = "./models/wordpiece/"
    tok_dir = "./tokenizers/wordpiece/"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    w = textwrap.TextWrapper(replace_whitespace=False, break_long_words=False, width=60, initial_indent='   ',
                             subsequent_indent='  ')


    def see(text):
        print('\n\033[3m' + '\n\n'.join(['\n'.join(w.wrap(line))
                                         for line in text.splitlines() if line.strip() != '']) + '\033[0m\n')


    models = []
    toks = []
    names = []

    for curr_dir in enumerate(os.listdir(model_dir)):
        name = curr_dir[1]

        model = AutoModelForCausalLM.from_pretrained(model_dir + name)

        models.append(model)
        names.append(name)

        tok_name = name.split("-")[1] + "-" + name.split("-")[2]

        # if "sp" in tok_name:
        #     tok = AlbertTokenizerFast.from_pretrained(tok_dir + tok_name + "/")
        # else:
        #     tok = RobertaTokenizerFast.from_pretrained(tok_dir + tok_name + "/")
        tok = AutoTokenizer.from_pretrained(tok_dir + tok_name + "/")

        tok.pad_token_id = tok.eos_token_id
        toks.append(tok)


    # Reading prompts from a file
    prompt_batches = []

    # Open and read the file
    with open('prompts.txt', 'r') as file:
        # Read all lines from the file
        prompts = file.readlines()

    # Strip any extra whitespace (like newlines) from each prompt
    prompts = [prompt.strip() for prompt in prompts]

    batch_factor = 64  # 4 or 128

    # Group the prompts into batches of 10
    for i in range(0, len(prompts), 4):
        # Slice the prompts list to get a batch of 10
        batch = prompts[i:i + 4]
        # Append the batch to the prompt_batches list
        prompt_batches.append(batch * batch_factor)

        # GQA names:
    # model_name1 = 'CUSTOM-GQA-1KV-KQV-1F-8.0M-3L-8H-384C-1024I-0.001lr'
    # model_name2 = 'CUSTOM-GQA-0.75KV-KQV-1F-8.0M-3L-8H-384C-1120I-0.001lr'
    # model_name2 = 'CUSTOM-GQA-0.5KV-KQV-1F-8.0M-3L-8H-384C-1216I-0.001lr'
    # model_name2 = 'CUSTOM-GQA-0.25KV-KQV-1F-8.0M-3L-8H-384C-1312I-0.001lr'
    # model_name2 = 'CUSTOM-GQA-0.125KV-KQV-1F-8.0M-3L-8H-384C-1360I-0.001lr'
    # KQV names:
    # model_name1 = 'CUSTOM-GQA-1KV-KQV-1F-8.0M-3L-4H-384C-1024I-0.001lr'
    # model_name2 = 'CUSTOM-GQA-1KV-KQV-16F-8.0M-3L-4H-384C-1744I-0.001lr'
    # model_name2 = 'CUSTOM-GQA-1KV-KQV-2F-8.0M-3L-4H-384C-1408I-0.001lr'
    # model_name2 = 'CUSTOM-GQA-1KV-KQV-4F-8.0M-3L-4H-384C-1600I-0.001lr'
    # model_name2 = 'CUSTOM-GQA-1KV-KQV-8F-8.0M-3L-4H-384C-1696I-0.001lr'
    # model1 = init_model(model_name=model_name1)
    # model2 = init_model(model_name=model_name2)
    avg_tokens_per_s, avg_chars_per_s = compare_models(models, toks, names, prompt_batches)

    def dump_dictionaries_to_file(avg_tokens_per_s, avg_chars_per_s, filename='sp_generation_test.json'):
        data = {
            'avg_tokens_per_s': avg_tokens_per_s,
            'avg_chars_per_s': avg_chars_per_s
        }
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

    # Example usage
    dump_dictionaries_to_file(avg_tokens_per_s, avg_chars_per_s)

    for i, model in enumerate(models):
        print(f"Model {names[i]} had {avg_tokens_per_s[i]} tokens/s and {avg_chars_per_s[i]} chars/s")

# ---------------------------------------
