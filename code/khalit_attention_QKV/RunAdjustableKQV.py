import textwrap 
w = textwrap.TextWrapper(replace_whitespace=False, break_long_words=False, width=60, initial_indent='   ', subsequent_indent='  ')
def see(text): print('\n\033[3m' + '\n\n'.join(['\n'.join(w.wrap(line))
                 for line in text.splitlines() if line.strip() != '']) + '\033[0m\n')
    
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoConfig, LlamaForCausalLM
from AdjustableKQV_GPTNeoForCausalLM import AdjustableKQV_GPTNeoForCausalLM
import torch

model_name = 'CUSTOM-KQV-2F-8.0M-3L-4H-384C-1408I-0.001lr'

tokenizer = AutoTokenizer.from_pretrained('10k-tok')
tokenizer.pad_token_id = tokenizer.eos_token_id

config = GPTNeoConfig.from_pretrained(f'models/{model_name}')
custom_model = AdjustableKQV_GPTNeoForCausalLM(config=config, kqv_size=config.hidden_size // 2)
trained_state_dict = torch.load(f'models/{model_name}/model_state.pt')
custom_model.load_state_dict(trained_state_dict)

prompt = 'Once upon a time, there was a chicken with'
input_ids = tokenizer.encode(prompt, return_tensors='pt')

output = custom_model.generate(
    input_ids,                              # input to the model
    max_length=300,                         # maximum generation length
    eos_token_id=tokenizer.eos_token_id,    # early stopping when eos_token is output

    # num_beams=1,                            # number of beams to use in generation
    temperature=1,
)
output_text = tokenizer.decode(output[0])

# textwrap with indentation on every new paragraph
see(output_text)
output