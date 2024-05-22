from datasets import load_dataset
from transformers import GPT2TokenizerFast
dataset = load_dataset('roneneldan/TinyStories')
dataset.save_to_disk("data/TinyStories")
tokenize_function = GPT2TokenizerFast.from_pretrained('./10k-tok', model_max_length=512)
if __name__ == "__main__":
    tokenized_dataset = dataset.map(
         lambda x: tokenize_function(x['text'], truncation=True, padding='max_length'),
         batched=True, num_proc=8, batch_size=1000)
    tokenized_dataset.save_to_disk(f'data/TokenizedTinyStories', num_proc=5)