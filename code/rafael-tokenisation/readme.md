This package contains a number of resources that allow the training and evaluation of language models:

**Evaluation Pipeline 2023**

This is the package that allows us to evaluate the trained models on both BLiMP and SuperGLUE tasks. It is based on the
original [BabyLM pipeline](https://github.com/babylm/evaluation-pipeline-2023), but includes a few changes to support
the different types of tokenizers.

Check [environment changes](#environment-changes)

**Gpt Generation Speed Test**

This is the package that measures the generative speed, in tokens per second and characters per second of decoder style
models

**Inference Time Test**

This is the package that allows us to evaluate the trained models on inference time. It is based on the
original [BabyLM pipeline](https://github.com/babylm/evaluation-pipeline-2023), but isolates a test of the inference
time.

Check [environment changes](#environment-changes)

**Model Training**

This is the set of scripts which can be used to train models with different tokenizers.

Check [environment changes](#environment-changes)

**Plotting**

This includes a notebook used to plot the evaluation of the language models

**Tokenizer Creation**

This includes a set of scripts and notebooks that allow the training of different tokenizers with different vocabulary
sizes. It also includes a notebook to retrieve characteristics of tokenizer such as average character per token.

### Environment changes

In order to train and evaluate the models with SentencePiece tokenizers, changes to the _Transformers_ package had to
be made. In the `env_changes` directory this changes can be retrieved and applied to your environment. These changes 
are different in train and evaluation, so chose the subdirectory accordingly. Keep in mind these changes are only useful
for models with SentencePiece tokenizers.