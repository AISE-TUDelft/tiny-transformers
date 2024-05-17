s document gives (1) a conceptual overview of the evaluation pipeline; (2) a practical guide to running the pipeline; and (3) how to publish your models. 

## The [BabyLM Eval Pipeline](https://github.com/babylm/evaluation-pipeline-2023)
The evaluation pipeline consists of the following components, some of which are additional tests that were released towards the end of the challenge to prevent excessive overfitting. 

1. **BLiMP** (Benchmark of Linguistic Minimal Pairs). This consists of pairs of minimally different sentences that contrast in **grammatical acceptability**. The original task contained 12 categories, but was extended with 5 supplementary tasks.  
2. **(Super)GLUE** (General Language Understanding Evaluation). A benchmark of 9 sentence(-pair) **fine-tuned language understanding** tasks. 


**Supplementary Tasks**
3. 5 supplementary BLiMP tasks.
4. **MSGS** (Mixed Signals Generalisation Set). #TODO
5. **AoA** (Age of Acquisition). #TODO 

> [!NOTE] Feel free to fill in the gaps here; I'm skipping some bits and labelling with #TODO. I think evaluating on these additional tests can likely give some novel insights to your models. 

I *try* to briefly describe BLiMP and GLUE below.

#### BLiMP
The original Benchmark of Linguistic Minimal Pairs ([Warstadt et al. 2019](https://arxiv.org/abs/1912.00582)) considers 12 grammatical categories, each specific in syntax, morphology, or semantics. Models are evaluated **zero-shot**, by comparing the probabilities of the sequences in a minimal pair, under the assumption that the acceptable sequence will be considered more likely than its unacceptable counterpart. 

| Phenomenon                    | Acceptable                                                         | Unacceptable                                                       |
| ----------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| **Anaphoric Agreement**       | Many girls insulted *themselves*.                                  | Many girls insulted *herself*.                                     |
| **Argument Structure**        | Rose wasn't *disturbing* Mark.                                     | Rose wasn't *boasting* Mark.                                       |
| **Binding**                   | Carlos said that Lori helped *him*.                                | Carlos said that Lori helped *himself*.                            |
| **Control/Raising**           | There was *bound* to be a fish escaping.                           | There was *unable* to be a fish escaping.                          |
| **Determiner-Noun Agreement** | Rachelle had bought that *chair*.                                  | Rachelle had bought that *chairs*.                                 |
| **Ellipsis**                  | Anne's doctor cleans one *important* book and Stacey cleans a few. | Anne's doctor cleans one book and Stacey cleans a few *important*. |
| **Fillter-Gap**               | Brett knew *what* many waiters find.                               | Brett knew *that* many waiters find.                               |
| **Irregular Forms**           | Aaron *broke* the unicycle.                                        | Aaron *broken* the unicycle.                                       |
| **Island Effects**            | Which *bikes* is John fixing?                                      | Which is John fixing *bikes*?                                      |
| **NPI Licensing**             | The truck has *clearly* tipped over.                               | The truck has *ever* tipped over.                                  |
| **Quantifiers**               | No boy knew *fewer than* six guys.                                 | No boy knew *at most* six guys                                     |
| **Subject-Verb Agreement**    | These casseroles *disgust* Kayla.                                  | These casseroles *disgusts* Kayla.                                 |

The BLiMP supplement includes five held-out categories covering dialogue and questions. You can find more info about these in [Section 5.1.1 of Warstadt et al. 2023](https://aclanthology.org/2023.conll-babylm.1/).

| **Phenomenon**                                    | **Acceptable Example**                                    | **Unacceptable Example**                                                                                 |
| ------------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Hypernym**                                      | If he is growing *herbs*, then he is growing *plants*.    | If he is growing *herbs*, then he is growing *basil*.                                                    |
| **Subject-Auxiliary Inversion**                   | Logan will go. Will logan go?                             | (this dataset is part of a private phd thesis that I cannot access; you can check it in the code though) |
| **Turn-Taking**                                   | David: Shouldn't *you* quit?<br>Sarah: No, *I* shouldn't. | David: Should *she* quit? <br>Sarah: No, *I* shouldn't.                                                  |
| **Question-Answer Congruence** (Easy)             | A: What did you purchase? <br>B: Bread                    | A: What did you purchase?<br>B: *David*.                                                                 |
| **QA Congruence** (Hard, with a distracting word) | When did you eat?<br>Several minutes ago.                 | When did you eat? <br>*Dinner*.                                                                          |
#### Super(GLUE) 
For some General Language Understanding Evaluation tasks, training data is plentiful, but for others it is limited or fails to match the genre of the test set. GLUE therefore favours models that can learn to represent linguistic knowledge in a way that facilitates *sample-efficient learning and effective cross-task knowledge transfer*. 

BabyLM uses the following 7/9 tasks from the original [GLUE paper (2018)](https://openreview.net/pdf?id=rJ4km2R5t7). The metric(s) used for the task are given as tags. If a task uses several metrics, they are averaged into one score before computing the macro average. 

- **Single-Sentence Tasks.**
	- **CoLA** (Corpus of Linguistic Acceptability): Given a sequence of words, is it grammatically acceptable? #mcc ()
	- **SST-2** (Stanford Sentiment Treebank): Given a movie review (1 sentence), predict whether its sentiment is positive/negative. #accuracy, 
- **Similarity and Paraphrase**
	- **MRPC** (Microsoft Research Paraphrase Corpus): Given sentence pairs from news sources, are they semantically equivalent? #accuracy #f1 
	- **QQP** (Quora Question Pairs): Given a pair of questions, are they semantically equivalent? #accuracy #f1 
- **Natural Language Inference**
	- **MNLI** (Multi-Genre Natural Language Inference Corpus): Given a premise and hypothesis, predict whether the premise entails the hypothesis, contradicts the hypothesis, or neither. #accuracy
	- **MNLI-mismatched**: MNLI sentence pairs are gathered from 10 domains. The above check *in*-domain, while mismatched checks *cross*-domain. #accuracy
	- **QNLI** (Stanford Question Answering Dataset): We first convert the task into sentence-pair classification by forming a pair between each question and wikipedia answer, and filtering out pairs with low lexical overlap. Given a context sentence and question, does the sentence contain the answer? #accuracy
	- **RTE** (Recognising Textual Entailment): Given two sentences, classify as *entailment* or not (also adopted in SuperGLUE as it tends to be harder). #accuracy (50.3)

And, they add an additional 3/8 tasks from the [SuperGLUE paper (2019)](https://arxiv.org/abs/1905.00537) paper. 

- **BoolQ**: answer a yes/no question about a short passage.  #accuracy (62.3)
- **MultiRC**: A paragraph and question about it are provided, and the model must predict which of a list of possible answers are correct (can be multiple). #F1 #exact-match (61.1/0.3) 
  #TODO: they do not compute #exact-match in the evaluation-pipeline, do they mention why in the paper; or does this mean we have to do it ourselves? Probably have to do it ourselves; see the [huggingface reference implementation](https://huggingface.co/spaces/evaluate-metric/super_glue/blob/main/super_glue.py). 
- **WSC** (Winograd Schema Challenge): Given a sentence with a pronoun, select the referrent of that pronoun from a list of choices. This is also present in the original GLUE as WNLI, though slightly modified to a 'natural language inference' problem. #accuracy (65.1)

I've added the most-frequent value in parantheses, which you can consider as a 'bottom-line' baseline (e.g. a model that always outputs 1). #TODO: These values are not given in the GLUE paper, but should be easily computable from the dataset itself. #TODO: This should also be possible to construct a baseline for BLiMP. 

#### MSGS
The Mixed-Signals Generalisation Set from [Warstadt et al. 2020](https://arxiv.org/abs/2010.05358) evaluates whether models learn *useful* representations: deeper linguistic embeddings are more useful than surface-level ones. This consists of 5 control and 6 ambiguous settings. 

#TODO: this section requires more work, and I have ommitted it from my `eval_baselines.py` script as I'm out of time. 

It only makes sense to compare a model’s preference between two features if it actually represents both features. This is the goal behind **control** experiments, in which we classify sentences based on a single linguistic or surface feature in a totally unambiguous setting. This is then used to compute a **linguistic bias score** (LBS) by running evaluations on fully ambiguous and partially datasets: if LBS is 1, the model shows a systematic linguistic bias; if it is -1, the model shows a systematic surface bias.

These are the tasks we fine-tune on. 
- "main_verb_control"
- "control_raising_control"
- "syntactic_category_control"
- "lexical_content_the_control"
- "relative_position_control"

- "main_verb_lexical_content_the"
- "main_verb_relative_token_position"
- "syntactic_category_lexical_content_the"
- "syntactic_category_relative_position"
- "control_raising_lexical_content_the"
- "control_raising_relative_token_position"

## Practical Information
Cool that's all the theory, honestly I skipped most of it; how do I actually run this stuff? 

#### Setup
Create a new environment for the pipeline, as there are a lot of external dependencies that may interfere with your development environment.

```sh
# create env, libmamba will save you years of waiting for conda
conda create -n babylm python --solver=libmamba 
conda activate babylm 

# you need to make sure you've fetched this submodule with git
# something like git submodule init, but check online
cd evaluation-pipeline 
pip install -e '.[dev]'
pip install torch 
pip install wandb

unzip filter_data.zip
```

In short, this is it; but check their [instructions](https://github.com/babylm/evaluation-pipeline) if you get stuck. These are some of the issues I ran into:

- If their torch version does not match your CUDA drivers, remove the version specification in `setup.py`. 
- On some (older) machines, `git-lfs` does not ship with `git`, and you may need to install it manually. 
- My docker container running `Ubuntu` did not have `gcc` installed either, which is necessary for the compilation of one of the dependencies (`sklearn`); install with `sudo apt install gcc`. 
- They use `python==3.10.12` in their [demo notebook](https://colab.research.google.com/drive/1HX2D3wztO81tKcqCeV_ecRcEUseBVuTc?usp=sharing); this does not work for my CUDA installation.
	Specifically, there is a [bug](https://discuss.pytorch.org/t/issues-on-using-nn-dataparallel-with-python-3-10-and-pytorch-1-11/146745/13) requiring you to change `lib/python3.10/site-packages/torch/cuda/nccl.py`, at line `51`: from `collections` to `collections.abc`. 

#### Running Evaluation
I've prepared a script called `evaluate.sh` that takes in as argument your model directory, and should handle the rest. It basically does the following: 

```sh 
# copy over the tokenizer (hf doesn't seem to follow symlinks, but it's only 0.5Mb)
cp 10k-tok/* results/models/{model_name}

# BLiMP
python babylm_eval.py 'path/to/model_and_tokenizer' decoder # encoder for BERT
# Super(GLUE)
./finetune_all_tasks.sh 'path/to/model_and_tokenizer' 

# Combine raw predictions into one file containing all tasks' predictions
# We don't care, as we just want to look at the computed scores for now.
# python collect_results.py 'path/to/model_and_tokenizer'
``` 

Collecting the results may pose a bit more work; we need to look in the model's directory and retrieve the `.json` scores per task and aggregate them appropriately as defined by each benchmark. 

I provide `eval_baselines.py` to do this; all you need to do is specify the `MODEL_DIR` of your models. This will parallelise evaluation (typically less GPU-intense; especially for BLiMP) and aggregate your results into one dictionary. 

#### Publishing Models
This should be as simple as:

```python
from transformers import AutoModel

model_name = 'GPT-11.0M-3L-4H-512C-1024I-0.0008lr'
model = AutoModel.from_pretrained(model_name)
model.push_to_hub(model_name)
```

