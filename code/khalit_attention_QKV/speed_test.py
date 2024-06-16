import torch
from transformers import AutoModelForCausalLM, AutoConfig, GPTNeoConfig, AutoTokenizer
from GQA_GPTNeoForCausalLM import CustomGPTNeoForCausalLM
from CustomGQA_GPTNeoConfig import CustomGPTNeoConfig
import re
import time
from memory_profiler import memory_usage
import textwrap
import os
import gc
import math
from AdjustableKQV_GPTNeoForCausalLM import AdjustableKQV_GPTNeoForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"
w = textwrap.TextWrapper(replace_whitespace=False, break_long_words=False, width=60, initial_indent='   ', subsequent_indent='  ')
def see(text): print('\n\033[3m' + '\n\n'.join(['\n'.join(w.wrap(line))
                 for line in text.splitlines() if line.strip() != '']) + '\033[0m\n')
    
AutoConfig.register("custom_gqa_gpt_neo", CustomGPTNeoConfig)  
AutoModelForCausalLM.register(CustomGPTNeoConfig, CustomGPTNeoForCausalLM)

tokenizer = AutoTokenizer.from_pretrained('10k-tok')
tokenizer.pad_token_id = tokenizer.eos_token_id

def init_model(model_name):
    # Define regex patterns
    pattern_kv = r'(\d+\.\d+)KV'  # Matches decimal numbers followed by 'KV'
    pattern_h = r'(\d+)H'  # Matches numbers followed by 'H'
    pattern_f = r'(\d+)F'  # Matches numbers followed by 'F'
    pattern_c = r'(\d+)C'  # Matches numbers followed by 'C'

    # Extract values using regex
    match_kv = re.search(pattern_kv, model_name)
    match_h = re.search(pattern_h, model_name)
    match_f = re.search(pattern_f, model_name)
    match_c = re.search(pattern_c, model_name)

    if match_kv is None:
        group_factor = 1
    else:
        group_factor = float(match_kv.group(1))
    num_heads = int(match_h.group(1))
    kqv_factor = float(match_f.group(1))
    embedding_size = int(match_c.group(1))

    num_kv_heads = int(num_heads * group_factor)
    kqv_size = int(embedding_size // kqv_factor)

    config_neo = GPTNeoConfig.from_pretrained(f'results/models_baseline/{model_name}')
    config_dict_1 = config_neo.to_dict()
    # Check if 'kqv_size' is present in config_dict and remove it if it is
    if 'kqv_size' in config_dict_1:
        del config_dict_1['kqv_size']
    # Check if 'num_kv_heads' is present in config_dict and remove it if it is
    if 'num_kv_heads' in config_dict_1:
        del config_dict_1['num_kv_heads']
    my_config = CustomGPTNeoConfig(num_kv_heads=num_kv_heads, kqv_size=kqv_size, **config_dict_1)

    custom_model = AutoModelForCausalLM.from_pretrained(
        f'results/models_baseline/{model_name}',
        config=my_config
    )

    return custom_model

def generate_batch(model, prompts, min_mem_usage):
    # print(f'Generating the batch for the model with {model.config.num_kv_heads} query groups and KQV factor {model.config.hidden_size // model.config.kqv_size}')
    gc.collect()  # Explicit garbage collection
    torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
    # Tokenize the prompts
    input_ids_list = [tokenizer.encode(prompt, return_tensors='pt') for prompt in prompts]

    # Determine the maximum sequence length
    max_length = max(len(ids[0]) for ids in input_ids_list)
    input_target_length = 32

    # Ensure that in case for some magical reason max_length is greater than the selected input target
    # we still use the maximum.
    max_length = max(max_length, input_target_length)
    # print("Input len: ", max_length)
    padded_input_ids = [torch.nn.functional.pad(ids, (0, max_length - len(ids[0])), 'constant', tokenizer.pad_token_id) for ids in input_ids_list]
    input_ids = torch.stack(padded_input_ids, dim=0).squeeze(1)
    # Stack the padded sequences to create a batch
    input_ids = torch.stack(padded_input_ids, dim=0).squeeze(1)

    # Move input_ids to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = input_ids.to(device)
    model.to(device)

    # Generate outputs in batch
    batch_size = input_ids.size(0)
    fixed_output_length = 256
    try:
        with torch.no_grad():
            outputs = []
            def generate():
                output = model.generate(
                    input_ids,
                    min_length=fixed_output_length,              # Set the minimum length
                    max_length=fixed_output_length,              # Set the maximum length as the same to insure fixed output length
                    eos_token_id=None,                    # Disable early stopping by EOS token
                    temperature=1,
                    do_sample=True                        # Enable sampling to allow for variability
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
        for output in outputs:
            decoded_output = tokenizer.decode(output, skip_special_tokens=True)
            output_texts.append(decoded_output)
            output_lengths.append(len(output))


        # for i, output_text in enumerate(output_texts):
        #     print(f"Prompt {i+1}: {prompts[i]}")
        #     print(f"Generated Text {i+1}:\n {output_text}")
        #     print(f"Number of Tokens {i+1}: {output_lengths[i]}")
        #     print("\n" + "-"*50 + "\n")
        # print("Number of outputs: ", len(output_lengths))

        # Return time taken per one token. 
        # Divide by output_lengths[0] because all outputs in a batch have the same length.
        # Divide by the number of outputs in a batch.
        return peak_memory_usage, time_taken / (fixed_output_length * len(outputs))

    except RuntimeError as e:
        print(f"Batch size {batch_size} failed due to: {e}")

def run_batches(model, prompt_batches, time_accumulator):
    warmup_prompts = [
        'Once upon a time, there was a guy named',
        'In a distant land, there was a mysterious forest',
        'The scientists in a school made',
        # Add more prompts as needed
    ]
    # Warm-up runs
    print("Warm-up runs:")
    for _ in range(3):
        generate_batch(model=model, prompts=warmup_prompts * 50, min_mem_usage=[None])
        print("Still warming up...")

    print("-----------------------------------------------------------------")

    min_mem_usage = [None]
    for prompt_batch in prompt_batches:
        memory1, time1 = generate_batch(model, prompt_batch, min_mem_usage=min_mem_usage)
        time_accumulator.append(time1)
        print("Memory received: ", memory1)
        print("Time received: ", time1)

def compare_models(model1, model2, prompt_batches):
    times1 = []
    times2 = []
    run_batches(model1, prompt_batches, times1)
    run_batches(model2, prompt_batches, times2)
    average_time1 = sum(times1) / len(times1)
    average_time2 = sum(times2) / len(times2)
    print("Average inference time for model 1 is: ", average_time1)
    print("Average inference time for model 2 is: ", average_time2)
    print("Model 2 is faster by: ", average_time1 - average_time2, " s per token")
    speedup = ((average_time1 / average_time2) - 1) * 100
    speedup = round(speedup, 2)
    print(f"Which is equivalent to {speedup}% speedup")
    return average_time1, average_time2, speedup


# Reading prompts from a file
prompt_batches = []

# Open and read the file
with open('prompts.txt', 'r') as file:
    # Read all lines from the file
    prompts = file.readlines()

# Strip any extra whitespace (like newlines) from each prompt
prompts = [prompt.strip() for prompt in prompts]

batch_factor = 128  # 4 or 128

# Group the prompts into batches of 10
for i in range(0, len(prompts), 4):
    # Slice the prompts list to get a batch of 10
    batch = prompts[i:i + 4]
    # Append the batch to the prompt_batches list
    prompt_batches.append(batch * batch_factor)         

# GQA names:
model_name1 = 'CUSTOM-GQA-1KV-KQV-1F-8.0M-3L-8H-384C-1024I-0.001lr'
# model_name2 = 'CUSTOM-GQA-0.75KV-KQV-1F-8.0M-3L-8H-384C-1120I-0.001lr'
# model_name2 = 'CUSTOM-GQA-0.5KV-KQV-1F-8.0M-3L-8H-384C-1216I-0.001lr'
# model_name2 = 'CUSTOM-GQA-0.25KV-KQV-1F-8.0M-3L-8H-384C-1312I-0.001lr'
model_name2 = 'CUSTOM-GQA-0.125KV-KQV-1F-8.0M-3L-8H-384C-1360I-0.001lr'
# KQV names:
# model_name2 = 'CUSTOM-GQA-0.5KV-KQV-1F-8.0M-3L-8H-384C-1216I-0.001lr'
# model_name2 = 'CUSTOM-GQA-1KV-KQV-2F-7.0M-3L-8H-384C-1216I-0.001lr'
# model_name2 = 'CUSTOM-GQA-1KV-KQV-2F-8.0M-3L-4H-384C-1408I-0.001lr'
# model_name2 = 'CUSTOM-GQA-1KV-KQV-4F-8.0M-3L-4H-384C-1600I-0.001lr'
# model_name2 = 'CUSTOM-GQA-1KV-KQV-8F-8.0M-3L-4H-384C-1696I-0.001lr'

# config1 = GPTNeoConfig.from_pretrained(f'results/models_baseline/{model_name1}')
# model1 = AdjustableKQV_GPTNeoForCausalLM(config=config1, kqv_size=config1.hidden_size)
# trained_state_dict = torch.load(f'results/models_baseline/{model_name1}/model_state.pt')
# model1.load_state_dict(trained_state_dict)
# # print(type(model1))

# config2 = GPTNeoConfig.from_pretrained(f'results/models_baseline/{model_name2}')
# model2 = AdjustableKQV_GPTNeoForCausalLM(config=config2, kqv_size=config2.hidden_size // 2)
# trained_state_dict = torch.load(f'results/models_baseline/{model_name2}/model_state.pt')
# model2.load_state_dict(trained_state_dict)

model1 = init_model(model_name=model_name1) 
model2 = init_model(model_name=model_name2)

times1 = []
times2 = []
for i in range(0,3):
    time1, time2, speedup = compare_models(model1=model1, model2=model2, prompt_batches=prompt_batches)
    times1.append(time1)
    times2.append(time2)
final_time_microseconds1 = 1000000 * (sum(times1) / len(times1))
final_time_microseconds2 = 1000000 * (sum(times2) / len(times2))
final_speedup = ((final_time_microseconds1 / final_time_microseconds2) - 1) * 100

final_speedup = round(speedup, 2)
final_time_microseconds1 = round(final_time_microseconds1, 3)
final_time_microseconds2 = round(final_time_microseconds2, 3)

print(f"FINAL AVERAGED TIME FOR THE BASELINE: {final_time_microseconds1} microseconds.")
print(f"FINAL AVERAGED TIME FOR MODEL {model_name2}: {final_time_microseconds2} microseconds.")
print(f"FINAL AVERAGED SPEEDUP: {final_speedup}%")
