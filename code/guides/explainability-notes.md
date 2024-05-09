# Explainability for LLMs NOTES

# What does explainability mean for LLMs?

Large Language Models (LLMs) such as BERT, GPT-4, or LLaMA-2 have demonstrated impressive performance across a wide range of NLP tasks. Since LLMs are notoriously complex “black-box ” systems, their inner working mechanisms are opaque, and the high complexity makes model interpretation much more challenging. 

> Explainability refers to the ability to explain or present the behavior of models in human-understandable terms.
> 

Improving the explainability of LLMs is crucial for two reasons:

1. **For general user purposes**, explainability builds appropriate trust by elucidating the reasoning mechanism behind the prediction in an understandable manner.
2. **For researchers and developers**, explaining the model’s behaviors provides insight to identify unintended biases, risks, and areas for performance improvement. It facilitates the ability to track model capabilities over time and make comparisons between models. Overall it acts as a debugging aid to quickly advance model performance improvements.

As models become larger, understanding and interpreting their decision-making processes becomes more difficult due to increased internal complexity and vastness of training data. Over the years LLM explainability has gained more and more interest among AI researchers. It is a field that is consistently expanding and there are several explainability approaches proposed by researchers. The keywords “LLM” and “explainability” were included in over 1.7k papers in 2023.

![Untitled](Explainability%20for%20LLMs%20NOTES%20eb11ae6931054b04aefac24ea93455e6/Untitled.png)

The aim of this presentation is to:

- showcase some of the newest approaches;
- discuss about how explainability can be leveraged;
- present current challenges and future trends.

# Taxonomy

Due the sudden growth of this field of research, there were proposed many explainability techniques that aims to trace and assess models’ behaviors. [Zhao et al.](https://arxiv.org/abs/2309.01029) propose a structured overview of methods for explaining Transformer-based language models. Approaches that aims to provide explanations for LLMs can be divided into two broad domains: local analysis and global analysis.

- The goal of **local analysis** is to explain how the LLM makes the prediction for a specific input. Considering a scenario where we have a language model and we input a specific text into the model. The model then produces an output, the role of explanation is to clarify the process by which the model generated that particular output.
- **Global analysis** aims to provide a broad understanding of how the LLM works overall. Understanding what the individual components (neurons, hidden layers, and larger modules) have encoded and explain the knowledge/linguistic properties learned by the individual components.

## Local analysis

### **Feature attribution explanation**

Aiming to quantify the relevance of each input token to a model’s prediction.

Given an input $x$ with $n$ tokens $\{ x_i, x_2, ..., x_n\}$ a pre-trained language model $f$ outputs $f(x)$. Attribution methods assign a relevance score $R(x_i)$ to each token $x_i$, reflecting its contribution to $f(x)$.

![Untitled](Explainability%20for%20LLMs%20NOTES%20eb11ae6931054b04aefac24ea93455e6/Untitled%201.png)

- **Gradient based methods.** They are a natural approach for feature attribution. It consists in computing per-token importance scores computing the backward gradient. A common technique is *input X gradient*.
- **Surrogate Models.** These methods use simpler, more human-comprehensible models to explain individual predictions of black-box models. Surrogate models include (decision trees, linear models, etc.). The explanation models need to satisfy additivity (i.e. the total impact of the prediction should equal the sum of the individual impacts of each explanatory factor.).
- **Perturbation based methods.** Such as LIME or SHAP, alter input features to observe changes in model output.

### Dissecting transformer blocks

Tracking transformer block’s components can provide rich information on its intermediate processing. In a transformer inference pass, the input embeddings are transformed through a seqeuence of L transformer layers, each composed of a multi-head self-attention sublayer and by a MLP sublayer. Formally the representation of the token **i** at layer **l** is obtained:

$$
x_i^l = x_i^{l-1}+a_i^l+m_i^l
$$

Where  $a_i^l$ and $m_i^l$ are the outputs from the l-th Multi Head Self-Attention (MHSA) and Multi Layer Perceptron (MLP) sublayers, respectively.

New studies are focusing on the interaction between these sublayers.

**Analyzing MHSA sublayers**

Attention mechanism in MHSA sublayers are instrumental in capturing meaningful correlations between intermediate states of input that can explain model’s predictions. 

![Untitled](Explainability%20for%20LLMs%20NOTES%20eb11ae6931054b04aefac24ea93455e6/Untitled%202.png)

Many studies have analyzed the linguistic capabilities of a transformer by tracking attention weights. Attention mechanism typically prioritize specific tokens while diminishing the emphasis on frequent words or special tokens.

[Kobayashi et al., 2020](https://arxiv.org/abs/2004.10102) have reformulated the attention mechanism as a weighted sum of vectors  between the transformed input vector $f(x_j)$ and its attention weight $\alpha_{i,j}$.

![Screenshot 2024-04-15 at 17.56.17.png](Explainability%20for%20LLMs%20NOTES%20eb11ae6931054b04aefac24ea93455e6/Screenshot_2024-04-15_at_17.56.17.png)

![Screenshot 2024-04-15 at 17.56.57.png](Explainability%20for%20LLMs%20NOTES%20eb11ae6931054b04aefac24ea93455e6/Screenshot_2024-04-15_at_17.56.57.png)

![Untitled](Explainability%20for%20LLMs%20NOTES%20eb11ae6931054b04aefac24ea93455e6/Untitled%203.png)

Where W are the attention weights and b are the biases.

They propose the measurement of the **norm of the weighted transformed vector** $||\alpha f(x)||$ to analyze the attention mechanism behavior. Their findings are: 

- (i) contrary to previous studies, BERT pays poor attention to special tokens, and
- (ii) reasonable word alignment can be extracted from attention mechanisms of Transformer.

### **Analyzing Feed-Forward sublayers**

More recently, a surge of work have investigated the knowledge captured by FFN layers. These layers are consuming the majority of each layer’s parameter budget $8d^2$ compared to $4d^2$ for self-attention layers (d represents the model’s hidden dimension). 

[Geva et al., 2020](https://arxiv.org/abs/2203.14680); reverse engineered the operation of the feed-forward network (FFN) layers, one of the building blocks of transformer models. 

![Untitled](Explainability%20for%20LLMs%20NOTES%20eb11ae6931054b04aefac24ea93455e6/Untitled%204.png)

They viewed the token representation as a changing distribution over the vocabulary, and the output from each FFN layer as an additive update to that distribution. Then, they analyzed the FFN updates in the vocabulary space, showing that each update can be decomposed to sub-updates corresponding to single FFN parameter vectors, each promoting concepts that are often human-interpretable. 

Feed-forward layers apply additive updates (A) to the token representation x, which can be interpreted as a distribution over the vocabulary (B). An update is a set of sub-updates
induced by parameter vectors $v_1$, ..., $v_{dm}$ (C), each can be interpreted as a concept in the vocabulary space (D). They show that leveraging this findings can help controlling LM predictions.

## Global analysis

Different from local explanations that aim to explain a model’s individual prediction, global explanations  offer insights into the inner workings of language models. Aiming to understand what the individual components have encoded and explain the knowledge or linguistic properties stored in the hidden state activations of a model. 

### **Probing-based explanations**

The probing technique refers to methods used to understand the knowledge that LLMs have captured. One of this techniques is the **classifier-based probing**, where a shallow classifier is trained on top of the pre-trained or fine-tuned model. 

Parameters of the models are first frozen, and the model generates representations for input words, phrases, or sentences and learns parameters like attention weights. These representations are fed into a **probe classifier**, whose task is to identify certain linguistic properties or reasoning abilities acquired by the model. 

An example are probing classifiers trained for identifying attention heads patterns, [Kovaleva et al., 2019](https://arxiv.org/pdf/1908.08593.pdf); [Clark et al., 2019](https://aclanthology.org/W19-4828.pdf) proposed a classifiers to identify patterns using self-attention maps sampled on random inputs, then prunes heads based on this to improve model efficiency. 

For a given input, they extracted self-attention weights for each head in every layer. This results in a 2D float array of shape L×L, where L is the length of an input sequence. We will refer to such arrays as **self-attention maps**. Analysis of individual self-attention maps allows
us to determine which target tokens are attended to the most as the input is processed token by token.

![Screenshot 2024-04-24 at 11.29.53.png](Explainability%20for%20LLMs%20NOTES%20eb11ae6931054b04aefac24ea93455e6/Screenshot_2024-04-24_at_11.29.53.png)

Manual inspection of self-attention maps for both basic pre-trained and fine-tuned BERT models suggested that there is a limited set of **self-attention map types** that are repeatedly encoded across different heads. 

After identifying this 5 common patterns they trained a neural network with 8 convolutional layers and ReLU activation functions to classify input maps into one of these classes. 

One example of usage for this classifier is estimating the proportion of different self-attention patterns for a set of target GLUE (General Language Understanding Evaluation) tasks. The figure shows the self attention map types distribution across GLUE tasks.

![Screenshot 2024-04-25 at 15.19.37.png](Explainability%20for%20LLMs%20NOTES%20eb11ae6931054b04aefac24ea93455e6/Screenshot_2024-04-25_at_15.19.37.png)

### Mechanistic Interpretability

Mechanistic interpretability seeks to comprehend language models by examining individual neurons and their interconnections, often conceptualized as circuits. This field encompasses various approaches, on of them is **circuit discovery**. The field focuses on reverse engineering state-of-the-art models by identifying circuits, subgraphs of networks consisting sets of tightly linked features and the weights between them.

In mechanistic interpretability, we want to understand the correspondence between the components of a model and human-understandable concepts. A useful abstraction for this goal is circuits. If we think of a model as a computational graph M where nodes are terms in its forward pass (neurons, attention heads, embeddings, etc.) and edges are the interactions between those terms (residual connections, attention, projections, etc.), a circuit C is a subgraph of M responsible for some behavior (such as completing the IOI task).

A prominent example of this approach is the analysis of GPT-2 small ([Wang et al., 2022](https://arxiv.org/pdf/2211.00593)), this study identified a human understandable subgraph within the computational graph responsible for performing the indirect object identification (IOI) task. 

A sentence containing indirect object identification (IOI) has an initial dependent
clause, e.g “When Mary and John went to the store”, and a main clause, e.g “John gave a bottle of milk to Mary”. The initial clause introduces the indirect object (IO) “Mary” and the subject (S)
“John”. The main clause refers to the subject a second time, and in all our examples of IOI, the
subject gives an object to the IO. The IOI task is to predict the final token in the sentence to be the indirect object “Mary”.

In IOI, sentences like “When Mary and John went to the store, John gave a drink to” are expected to be completed with “Mary”. The study discovered a circuit comprising 26 attention heads – just 1.1% of the total (head, token position) pairs – that predominantly manages this task.

![Screenshot 2024-04-25 at 16.55.06.png](Explainability%20for%20LLMs%20NOTES%20eb11ae6931054b04aefac24ea93455e6/Screenshot_2024-04-25_at_16.55.06.png)

Their circuit contains three major classes of heads, corresponding to the three steps of the algorithm above:

- **Duplicate Token Heads** identify tokens that have already appeared in the sentence. They are active at the S2 token, attend primarily to the S1 token, and signal that token duplication has occurred by writing the position of the duplicate token.
- **S-Inhibition Heads** remove duplicate tokens from Name Mover Heads’ attention. They are active at the END token, attend to the S2 token, and write in the query of the Name Mover Heads, inhibiting their attention to S1 and S2 tokens.
- **Name Mover Heads** output the remaining name. They are active at END, attend to previous names in the sentence, and copy the names they attend to. Due to the S-Inhibition Heads, they attend to the IO token over the S1 and S2 tokens.

# Leveraging Explainability

Explainability can be used as a tool to debug and improve models.

## Model editing

In recent years, there has been a surge in techniques for editing LLMs. The goal is to efficiently modify the knowledge or behavior of LLMs within specific domains without adversely their performance on other inputs.

**Locate-Then-Edit** paradigm first identifies the parameters corresponding to the specific knowledge and then modifies them by directly updating the target parameters. 

[Dai et al., 2022](https://arxiv.org/pdf/2104.08696) introduces a knowledge attribution technique to pinpoint the knowledge neuron (a key-value pair in the FFN matrix) that embodies the knowledge and then updates these neurons. 

![Screenshot 2024-04-26 at 14.25.12.png](Explainability%20for%20LLMs%20NOTES%20eb11ae6931054b04aefac24ea93455e6/Screenshot_2024-04-26_at_14.25.12.png)

## Enhancing model capability

While LLMs demonstrate versatility in various NLP tasks, insights from explainability can significantly enhance these capabilities.

In-context learning (ICL) is a method of prompt engineering where the model is shown task demonstrations as part of the prompt in natural language. Using ICL, you can utilize pre-trained large language models (LLMs) to solve new tasks without fine-tuning. ICL offers a promising approach to harness the full potential of LLMs. Despite its significance, the inner working mechanism of ICL remains an open question.

Through mechanistic interpretability [Wang et al., 2023](https://aclanthology.org/2023.emnlp-main.609/) reveal that **label words** in the demonstration examples function as anchors that aggregate and distribute information
in ICL.

In shallow layers, label words gather information from demonstrations to form semantic representations for deeper processing, while deep layers extract and utilize this information from
label words to formulate the final prediction.

![Screenshot 2024-04-29 at 10.44.38.png](Explainability%20for%20LLMs%20NOTES%20eb11ae6931054b04aefac24ea93455e6/Screenshot_2024-04-29_at_10.44.38.png)

Those findings were used to improve ICL performance with an anchor re-weighting method.

## Controllable Generation

Though large language models have obtained superior performance in text generation, they sometimes fall short of producing factual content. Leveraging explainability provides opportunities for building inference-time and fast techniques to improve generation models’ factuality, calibration, and controllability and align more with human preference.

Hallucinations in LLMs refer to generated content not based on training data or facts, various factors such as imperfect learning and decoding contribute to this. To mitigate hallucinations, initial approaches used reinforcement learning from human feedback (RLHF).

Leveraging explainability provides a significantly less expensive way to reduce hallucination, enjoying the advantage of being adjustable and minimally invasive.

[Li et al., 2023b](https://arxiv.org/abs/2306.03341) uses a probing based classifier to locate and find “truthful” heads and directions; showing that there is  an interesting pattern of specialization across attention heads. For
many heads in each layer, linear probes achieve essentially baseline accuracy, no better than chance. However, a significant proportion display strong performance. 

![Screenshot 2024-04-29 at 12.01.32.png](Explainability%20for%20LLMs%20NOTES%20eb11ae6931054b04aefac24ea93455e6/Screenshot_2024-04-29_at_12.01.32.png)

Information is mostly processed in early to middle layers and that a small portion of heads stands out in each layer.

They propose **inference-time intervention** (ITI), a computationally inexpensive strategy to intervene on the attention head to shift the activations in the “truthful” direction, which achieves comparable or better performance toward the instruction-fine-tuned model.

![Screenshot 2024-04-29 at 11.44.13.png](Explainability%20for%20LLMs%20NOTES%20eb11ae6931054b04aefac24ea93455e6/Screenshot_2024-04-29_at_11.44.13.png)

# Current challenges and future trends

**Explaining without ground truth**

Ground truth explanations for LLMs are usually inaccessible. For example, there are currently no benchmark datasets to evaluate the global explanation of individual components captured by LLMs. This presents two main challenges. First, it is difficult to design explanation algorithms that accurately reflect an LLM’s decision-making process. Second, the lack of ground truth makes evaluating explanation faithfulness and fidelity problematic. It is also challenging to select a suitable explanation among various methods in the absence of ground truth guidance. Potential solutions include involving human evaluations and creating synthetic explanatory datasets.

**Sources of emergent abilities**

LLMs exhibit surprising new capabilities as the model scale and training data increases, even without being explicitly trained to perform these tasks. Elucidating the origins of these emergent abilities remains an open research challenge, especially for proprietary models like ChatGPT and Claude whose architectures and training data are unpublished.

- Model perspective:
    - What specific model architectures give rise to the impressive emergent abilities of LLMs?
    - What is the minimum model complexity and scale needed to achieve strong performance across diverse language tasks?
- Data perspective:
    - Which specific subsets of the massive training data are responsible for particular model predictions, and is it possible to locate these examples?
    - Are emergent abilities the result of model training or an artifact of data contamination issues?
    - Are training data quality or quantity more important for effective pre-training and fine-tuning of LLMs?

**Attention redundancy**

Recent research has investigated attention redundancy using interpretability techniques in large language models for both traditional fine-tuning and prompting paradigms. It was found ([Bian et al., 2021](https://aclanthology.org/2021.naacl-main.72/)) that there is redundancy in both attention heads and feedforward networks. These findings suggest that many attention heads and other components are redundant. This presents opportunities to develop model compression techniques that prune redundant modules while preserving performance on downstream tasks.

**Shifting from Snapshot Explainability to Temporal Analysis**

There is also an viewpoint that current interpretability research neglect the training dynamics. Existing research is mainly post-hoc explanation on fully trained models. The lack of developmental investigation on training process can generate biased explanation by failing in targeting emerging abilities or vestigial parts that convergence counts on, namely phase transitions.

By examining several checkpoints during training, [Chen et al. (2023a)](https://arxiv.org/pdf/2309.07311v4) identified an abrupt pre-training window wherein models gain Syntactic Attention Structure (SAS), which occurs when a specialized attention head focus on a word’s syntactic neighbors, and meanwhile a steep drop in training loss. They also showed that SAS is critical for acquiring grammatical abilities during learning.

Inspired by such a perspective, development analysis could uncover more casual relations and training patterns in the training process that are helpful in understanding and improving model performance.