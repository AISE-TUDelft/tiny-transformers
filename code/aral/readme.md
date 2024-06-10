
## Creating a New `GPT-Neo` 
I detail my process for modifying the `GPT-Neo` architecture to use RoPE instead of normal PE. Then, I train on TinyStories, and modify the BabyLM evaluation to work with the custom model. 

#### 1. Modifying `GPT-Neo` Architecture 

1. **Copy `modeling_gpt_neo.py` from the `transformers` library so we can modify it without affecting the rest of the library**. This requires some re-naming of imports. I'm not touching the `GPTNeoFlashAttention2` module, our models are small enough we do not need to worry about attention optimisation. 

2. **Modify `self.wpe` in `GPTNeoModel` to use `RoPE` from `GPT-NeoX`**. This involved inserting some RoPE computations in the `GPTNeoSelfAttention.__init__()` and its `_attn()` method. Also remember to remove the original default PEs. 

3. **Add relevant fields to `GPTNeoConfig`**. These fields are added, where `rope_scaling` is explained [here](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/).

    ```python 
    config.rotary_pct          :float = 0.25   # percentage of hidden dims to allocate to RoPE
    # config.rotary_ndims   # computed as head_size * rotary_pct
    config.rotary_emb_base       :int = 10_000 # frequency base
    config.rope_scaling.type     :str = 'linear' | 'dynamic' | None
    config.rope_scaling.factor :float = 2.0
    ```

#### 2. Training 
I want to try out two context-length configurations of `GPT-Neo` with RoPE: `512` like the other baselines, and `64` to investigate generalisability when the context window is smaller than input sequences. 

###### `512 context length`
This can use the existing tokenised dataset.

###### `64 context length`
We need to train for $512 / 64 = 8$ more samples as we now only have `64` 'actual' samples with the triangular training mask for GPT models. 

An alternative would be to feed the `512 context length` samples into the model, and modify the `forward` function to iterate over them in segments of `64`. But, I leave this for future work. 

Noticing there are three flavours of RoPE already implemented in `GPT-NeoX`, I also want to try those. 

###### RoPE Variations
There exist actually three variations of RoPE in `GPT-NeoX`: with `DynamicNTKScaling` , `LinearScaling` used in Llama, and a third version which I assume is without scaling. 

#### 3. Evaluation 
Run BabyLM'23 evaluation with two configurations: 

1. `512` and `64 context length`
2. `64 context length` model fine-tuning on sequences of `512 context length`, to test whether its context-generalisation holds up.


