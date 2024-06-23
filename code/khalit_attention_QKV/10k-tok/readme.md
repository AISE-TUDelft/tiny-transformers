## Tokenizer Fix for Interop between `RoBERTa` and `GPTNeo`

When instantiating a `RobertaModel` from our `config` with its `pad_token_id` specified from the `10k-gpt-neo` tokenizer, we run into an issue. On initialising positional embeddings, 

```python
# modeling_roberta.py:85
self.position_embeddings = nn.Embedding(
	config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
)
```

We get an error that 

```python 
AssertionError: Padding_idx must be within num_embeddings
```

Looking through the code reveals that `num_embeddings` here does not refer to the vocabulary size (i.e. the size of the embedding table), but rather to the number of positional embeddings. We have only `512` positional embeddings, while we have `10,000` tokens in our vocabulary. 

Reading the documentation for `nn.Embedding` tells us that the `padding_idx` field is optionally provided to avoid computing gradients for padding tokens. I.e. we use a fixed 'pad' vector of all zeroes throughout training, and do not update it with gradient descent. 

This leaves two possible solutions:

1. Reindex the padding token in the tokenizer to be within `num_embeddings`. We need to make sure that we swap the token at that location with the index of the padding token. 
2. As shown in the `nn.Embedding` documentation, the `padding_idx` field is optional (and is also not used in our `GPT-Neo` implementation). If we omit it, the `RoBERTa` model will just apply a learned positional embedding to this token and probably get away with it. 

The padding token is one of several [[#special tokens]]. Let's first understand these so we don't mess anything up; then, we can revisit our solutions. 

#### Special Tokens 
Last week we also tried instantiating the tokeniser as a `RobertaTokenizer` or a `GPTNeoTokenizer`:

```python 
gpt_tok = GPT2TokenizerFast.from_pretrained('10k-gpt-neo', model_max_length=config.hidden_size)
rob_tok = RobertaTokenizer.from_pretrained('10k-gpt-neo', model_max_length=config.hidden_size)
```

If you print these, you'll notice that they have differing special tokens. When fine-tuning a model, it's important you keep track of these as well. 

```python
RobertaTokenizer(... added_tokens_decoder={ 
	9999: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True), 
	10000: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True), 
	10001: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True), 
	10002: AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=False, special=True), 
})

GPT2TokenizerFast(... added_tokens_decoder={ 
	9999: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True), 
})
```

**What are special tokens even?** Let's start with the GPT case. We simply have a *padding* token to ensure that every sequence we pass to the model is of the same length (`512` in our model as given by `max_position_embeddings`). The padding token is denoted as `<|endoftext|>` above, but could also be `<pad>` or any other textual representation of the token. It is only there for when we detokenise a sequence, we can see the padding tokens actually represented in the text. 

**Why does RoBERTa have more special tokens?** (checking the `RoBERTa` paper, they do not make any modifications to the special tokens). This is explained in the [BERT paper](https://arxiv.org/abs/1810.04805) under Section 3. In short, it's to have unambiguous separation between two input sequences on multi-sentence classification tasks (e.g. question-answer). GPT doesn't have this as it is usually trained for only next-token prediction tasks (*causal* language modelling), instead of classification. 

> To make BERT handle a variety of down-stream tasks, our input representation is able to unambiguously represent both a single sentence and a pair of sentences (e.g., $\langle \text{Question, Answer} \rangle$ in one token sequence. [...] The first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token ([SEP]). Second, we add a learned embedding to every token indicating whether it belongs to sentence A or sentence B. 

**Okay, but what is the `<mask>` token for then?** This comes about from the masked-language modelling objective of BERT, as explained under Section 3.1. In short, the `<mask>` token is only seen during pre-training. In our case, we need to make sure this is part of the tokeniser when pre-training our `RoBERTa` models. 

> the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary, as in a standard LM. In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random. In contrast to denoising auto-encoders (Vincent et al., 2008), we only predict the masked words rather than reconstructing the entire input.

> Although this allows us to obtain a bidirectional pre-trained model, a downside is that we are creating a mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning. To mitigate this, we do not always replace “masked” words with the actual [MASK] token. The training data generator chooses 15% of the token positions at random for prediction. If the i-th token is chosen, we replace the i-th token with (1) the [MASK] token 80% of the time (2) a random token 10% of the time (3) the unchanged i-th token 10% of the time. Then, Ti will be used to predict the original token with cross entropy loss. We compare variations of this procedure in Appendix C.2.

#### Revisiting Solutions
We now know we need to ensure that the `RoBERTa` models we train have access to the necessary `<s>`, `</s>`, and `<mask>` tokens, besides the padding token that is also present in the `GPTNeo` tokeniser. 

While at first it seems like the padding token must be within the first 512 positions of the tokenizer, inspecting `RobertaEmbeddings` reveals something the authors did not mention about their implementation: In lines `92–96`, they assume the padding token should be at index 0, as all positions are computed relative to the padding token. This means that we actually *need 1 extra positional embedding*, even though we don't use the padding token's positional embedding (it's fixed to a zero vector). 

In actuality, the RoBERTa authors set the padding token to `index 1`. Not sure why, but this means that the positional encoding at `index 0` will not be used (and is thus a vector initialised properly but never trained). To check this, I loaded both the `RoBERTa-base` and its tokenizer, and indeed, the model has two more positional embeddings than the tokenizer specifies. Thus, let's use `index 0` for the padding token; as we actually want to use the weights that we initialise. 

#### The Solution
- For consistency with the TinyStories paper, let's use their tokenizer: `GPT2Tokenizer`. @Rafael, you should check which dataset this has been trained on. 
- For interoperability with RoBERTa, let's set the padding token to `index 0` (see below).
- As RoBERTa relies on the presence of the `<s>`, `</s>`, and `<mask>` special tokens, let's make sure we add these into the tokenizer as well. There is also the `<unk>` token for representing unknown characters, which we need to add as well. 
	- `<pad>`, `<s>`, `</s>`, and `<unk>` are placed at the start, at `index 0`, `index 1`, `index 2`, and `index 3`. The existing tokens at these locations are moved to the end of the tokenizer, to positions `9995`, `9996`, `9997`, and `9998` (replacing the tokens at those positions; and removing those last three merges from `merges.txt`)
	- `<mask>` is placed at `9999`, also replacing the existing value at that location, as it is the very last token in the `RoBERTa` tokenizer too. 

