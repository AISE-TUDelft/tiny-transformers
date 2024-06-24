import wandb
# wandb.login(key="6f46f55bd51d76400f1e877ea7dfa75c5c7d05d6")

from transformers import GPT2TokenizerFast, GPTNeoForCausalLM, GPTNeoConfig, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
from transformers import RobertaForCausalLM

from tokenizers import Tokenizer, pre_tokenizers, decoders, AddedToken, normalizers, trainers
from tokenizers.normalizers import BertNormalizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, \
                                WordPieceTrainer, UnigramTrainer

from tokenizers.implementations import SentencePieceUnigramTokenizer

from tokenizers.processors import RobertaProcessing, TemplateProcessing
from tqdm import tqdm

from tokenizers.pre_tokenizers import Whitespace

import sentencepiece as spm

if __name__ == "__main__":
    # Define the initial alphabet as a string
    initial_alphabet = set("".join(["!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿ÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞàáâãäåæçèéêëìíîïðĉČĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃ "]))
    cleaned_file = 'cleaned_training_data.txt'

    for e in [(2999, "3k"), (5999, "6k"), (9999, "10k"), (14999, "15k"), (19999, "20k")]:
        vocab_size = e[0]
        name = e[1]

        print("Doing tokenizatino for size: ", vocab_size)

        spm.SentencePieceTrainer.train(
            input=cleaned_file,
            model_prefix='./tokenizers/' + name + "-sp/sp_unigram",
            vocab_size=vocab_size,  # Adjust the vocabulary size as needed
            model_type='unigram',
            character_coverage=1.0,
            input_sentence_size=10000000,
            shuffle_input_sentence=True,
            # user_defined_symbols=list(initial_alphabet),
            pad_id=0,  # Padding ID (default is 0)
            unk_id=1,  # Unknown token ID (default is 1)
            bos_id=2,  # Beginning of sentence ID (default is 2)
            eos_id=3,  # End of sentence ID (default is 3)
            # user_defined_symbols="<mask>"
        )