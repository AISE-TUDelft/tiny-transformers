## Tiny Transformers

This repository contains both the code and the papers for the Q4/2024 Bachelor Research Project: **Architectural Decisions for Language Modelling with (Small) Transformers**. 

- [Zotero Group](https://www.zotero.org/groups/5467626/tiny_transformers) for managing our (shared) references.
- [Mattermost Channel](https://mattermost.tudelft.nl/cse3000-2324q4/channels/tiny-transformers) for communication.
- [Slide Deck](https://www.icloud.com/keynote/0487EgSVm26sNx1G3LGzfRtfw#Tiny_Transformers) for weekly meetings.


#### Overleaf links 

> [!NOTE]
> Please add your Overleaf links here once you have set them up ([guide](papers/readme.md)).
* Filip's [Link](https://www.overleaf.com/read/bfkdnmbgkjdv#65d3fa)
* Rafael's Overleaf [Link](https://www.overleaf.com/read/vcdhpwpgtnfg#280270)


--- 

> Description from ProjectForum

## Architectural Decisions for Language Modelling with (Small) Transformers

#### Prerequisites
- Motivation to learn (i.e. read papers) about state-of-the-art natural-language modelling techniques.
- Strong Python programming skills.
- Comfortable with shell environments (for submitting jobs to the supercomputer).
- You will need to remember some of your Linear Algebra and Calculus courses (matrices and gradient descent).
- Knowledge of deep learning libraries, like [`transformers`](https://huggingface.co/docs/transformers/index), [`pytorch`](https://pytorch.org/), [`tensorflow`](https://www.tensorflow.org/), is a big plus!

#### Introduction
<!-- Motivation -->
Language models (LMs) based on the transformer architecture [1] have, well, transformed the language processing domain. State-of-the-art large LMs are exposed to several orders of magnitude more data than a human, yet are certainly not leveraging all of this information. Given the current trend of exponentially-increasing model sizes, we are predicted to run out of high-quality textual data by 2026, which LMs tend to perform best on [2]. 

<!-- Aim --> 
This project studies the effect of architectural decisions on *small* LMs, with the overarching goal of increasing their sample-efficiency and minimising their parameter count. Recent studies [3, 4] have shown smaller models (≤33M parameters) can exhibit equivalent language understanding of their larger counterparts, and similarly display the desired emergent properties (e.g. reasoning and creativity) that drive their prevalence. This makes them ideal for exploring architectures in a compute-limited setting; and their small scale makes individual components more interpretable [3]. Additionally, small LMs allow local deployment, leading to better support for privacy, personalisation, and democratisation.

<!-- Goal --> 
Current research lacks precise understanding of how architectural decisions influence natural language understanding and fine-tuned task performance in small LMs. While Eldan & Li [3] show that small LMs can exhibit linguistic understanding greater than GPT-2 (125M parameters) by reducing the breadth of their training data, they miss quantitative evaluations in a down-stream, applied setting. Warstadt et al. [4] find that architectural optimisations are the most promising research direction for small LMs, but these decisions are rarely surveyed across different models. Moreover, research that applies LMs to downstream tasks rarely considers hyperparameters beyond the default, let alone architectural decisions [5]. 

###### Introducing RQ Terms
This project studies small LMs, particularly the impact of architectural decisions in the transformer blocks that perform the bulk of information integration. A block takes as input a sequence of embedded tokens, each represented by a vector of length defined by the model's hidden size. These inputs are transformed by the self-attention module, followed by a feed-forward network (FFN). The attention mechanism [6] itself consists of a number of heads, each of which match tokens' queries against other tokens' keys, to integrate information from the latter with the former. The FFNs (multi-layer perceptrons), allow the model to learn more complex, non-linear patterns. While this is the motivation for their introduction, it is worth noting that the exact way these two modules combine to form complex generated text is still not fully understood.

#### Goal
We explore small transformer architectures, to give context behind design decisions and their impact on language understanding in downstream tasks. 

#### Research Questions for the Sub-Projects
Following are some preliminary research questions based on the small LM exploration by Eldan & Li [3], though students are encouraged to construct/extend their own within the same domain. The models considered for each RQ are [GPT-Neo](https://huggingface.co/docs/transformers/model_doc/gpt_neo) and [BERT](https://huggingface.co/docs/transformers/model_doc/bert).

1. How is model performance affected by the width of hidden layers? [3]
2. How is model performance affected by the depth of its FFNs? [3]
3. How is model performance affected by the number of attention heads? [3, 7]
4. How is model performance affected by the number of transformer blocks, and the ordering of modules within them? [3, 8] 
5. How is model performance affected by the width of Query-Key vectors? [6]
 
Additionally, each student is expected to perform an interpretability study akin to [3] on the specific component of their RQ.
 
#### Approach
- **Base Model**: [GPT-Neo](https://huggingface.co/docs/transformers/model_doc/gpt_neo) & [BERT](https://huggingface.co/docs/transformers/model_doc/bert) at 9M parameters (4 blocks, 512 hidden size, 1024 intermediate FFN size, and 10k token vocabulary). Students augment this base model along the dimension of their RQ.
- **Dataset**: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), consisting of ~2M short stories written by GPT-3.5 and GPT-4.
- **Evaluation**: [BabyLM evaluation pipeline](https://github.com/babylm/evaluation-pipeline), consisting of [BLiMP](https://github.com/alexwarstadt/blimp) grammar test-suite and fine-tuning on [SuperGLUE](https://super.gluebenchmark.com/) tasks to evaluate downstream performance. We further study the total training time, and total training samples each model sees to draw conclusions about sample-efficiency. Lastly, we study the resulting models' size and inference speed to evaluate them in an edge-device context.
- **Hardware**: Due to their small size, the models can be pre-trained locally on students' laptops. Students are also able to train on the DelftBlue cluster. 

#### Q&A Sessions
- Wednesday 13 March, 16:00 ([Teams link](https://teams.microsoft.com/l/meetup-join/19%3ameeting_YjNhZWU5MjQtMzE0ZS00ZGM3LWE5OWUtMWY4ZTc1MjVkM2Vm%40thread.v2/0?context=%7b%22Tid%22%3a%22096e524d-6929-4030-8cd3-8ab42de0887b%22%2c%22Oid%22%3a%22439f835))
- Friday 15 March, 15:00 ([Teams link](https://teams.microsoft.com/l/meetup-join/19%3ameeting_NmM5NTNlNmQtM2FjNy00ZmYzLTlkYmItYzllMGYxMWU3ZTFk%40thread.v2/0?context=%7b%22Tid%22%3a%22096e524d-6929-4030-8cd3-8ab42de0887b%22%2c%22Oid%22%3a%22439f835))

---

#### Background
LLMs' increasingly large scale stimulates increasingly more research into optimisation. One strand is to take the models as they are, and optimise them in a top-down fashion. Another strand, perhaps most interestingly, consists of a bottom-up approach: experimenting with design decisions before and during pre-training. We further motivate the need for architectural optimisation by noting that such bottom-up advances are relatively unexplored in applied settings, compared to top-down approaches like prompt-engineering. The studies by Eldan & Li [3], and Warstadt et al. [4] stimulate research in this area by highlighting the competitive performance of small LMs in resource-constrained settings. 

TinyStories [3] is synthetic dataset of short childrens' stories generated by GPT-3.5 and GPT-4, consisting of ~2M short stories. By limiting the breadth of the dataset in this manner, it can be used to train and evaluate LMs with ≤33M parameters; and, these small LMs generate more coherent and grammatically-correct text than a 125M-parameter GPT-2 model. Additionally, these smaller models are more interpretable, and reveal individual functions for specific neurons (e.g. attending to the protagonist in a story). 

BabyLM [4], is a communal challenge in which participants competed to optimise training on a fixed data budget. The authors provided two budgets: a dataset of 10M and 100M words, similar both in content and amount of language a 12-year old is exposed to. This has led to some fruitful developments in curriculum learning (gradually increasing the complexity of training tasks), knowledge distillation (training a small student on the outputs of a larger teacher model), and architecture optimisation.

#### Recommended Material for Enthusiastic Students
- [Andrej Karpathy's series on Modeling GPT from scratch](https://www.youtube.com/@AndrejKarpathy/videos) (OpenAI co-founder).
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar.

### References
1. A. Vaswani _et al._, ‘Attention Is All You Need’. arXiv, Dec. 06, 2017. Accessed: Jan. 25, 2024. [Online]. Available: [http://arxiv.org/abs/1706.03762v5](http://arxiv.org/abs/1706.03762v5)
2. P. Villalobos, J. Sevilla, L. Heim, T. Besiroglu, M. Hobbhahn, and A. Ho, ‘Will we run out of data? An analysis of the limits of scaling datasets in Machine Learning’. arXiv, Oct. 25, 2022. Accessed: Jan. 23, 2024. [Online]. Available: [http://arxiv.org/abs/2211.04325](http://arxiv.org/abs/2211.04325)
3. R. Eldan and Y. Li, ‘TinyStories: How Small Can Language Models Be and Still Speak Coherent English?’ arXiv, May 24, 2023. Accessed: Nov. 01, 2023. [Online]. Available: [http://arxiv.org/abs/2305.07759](http://arxiv.org/abs/2305.07759)
4. A. Warstadt _et al._, ‘Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora’, in _Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning_, Singapore: Association for Computational Linguistics, 2023, pp. 1–6. doi: [10.18653/v1/2023.conll-babylm.1](https://doi.org/10.18653/v1/2023.conll-babylm.1).
5. A. Wettig, T. Gao, Z. Zhong, and D. Chen, ‘Should You Mask 15% in Masked Language Modeling?’ arXiv, Feb. 10, 2023. Accessed: Jan. 24, 2024. [Online]. Available: [http://arxiv.org/abs/2202.08005](http://arxiv.org/abs/2202.08005)
6. D. Bahdanau, K. Cho, and Y. Bengio, ‘Neural Machine Translation by Jointly Learning to Align and Translate’. arXiv, May 19, 2016. Accessed: Jan. 25, 2024. [Online]. Available: [http://arxiv.org/abs/1409.0473](http://arxiv.org/abs/1409.0473)
7. P. Michel, O. Levy, and G. Neubig, ‘Are Sixteen Heads Really Better than One?’ arXiv, Nov. 04, 2019. Accessed: Jan. 21, 2024. [Online]. Available: [http://arxiv.org/abs/1905.10650](http://arxiv.org/abs/1905.10650)
8. S. Shleifer, J. Weston, and M. Ott, ‘NormFormer: Improved Transformer Pretraining with Extra Normalization’. arXiv, Nov. 01, 2021. Accessed: Jan. 25, 2024. [Online]. Available: [http://arxiv.org/abs/2110.09456](http://arxiv.org/abs/2110.09456)


---
#### Related Reading
- J. Hoffmann _et al._, ‘Training Compute-Optimal Large Language Models’. arXiv, Mar. 29, 2022. Accessed: Jan. 25, 2024. [Online]. Available: [http://arxiv.org/abs/2203.15556](http://arxiv.org/abs/2203.15556)
- S. Gunasekar _et al._, ‘Textbooks Are All You Need’. arXiv, Oct. 02, 2023. Accessed: Jan. 23, 2024. [Online]. Available: [http://arxiv.org/abs/2306.11644](http://arxiv.org/abs/2306.11644)

- L. G. G. Charpentier and D. Samuel, ‘Not all layers are equally as important: Every Layer Counts BERT’. arXiv, Nov. 07, 2023. Accessed: Jan. 25, 2024. [Online]. Available: [http://arxiv.org/abs/2311.02265](http://arxiv.org/abs/2311.02265)
- R. D. Martinez _et al._, ‘CLIMB – Curriculum Learning for Infant-inspired Model Building’, in _Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning_, Singapore: Association for Computational Linguistics, 2023, pp. 84–99. doi: [10.18653/v1/2023.conll-babylm.10](https://doi.org/10.18653/v1/2023.conll-babylm.10).
- V. Sanh, L. Debut, J. Chaumond, and T. Wolf, ‘DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter’, Oct. 2019, doi: [10.48550/arXiv.1910.01108](https://doi.org/10.48550/arXiv.1910.01108).

- J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, ‘BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding’. arXiv, May 24, 2019. Accessed: Jan. 25, 2024. [Online]. Available: [http://arxiv.org/abs/1810.04805](http://arxiv.org/abs/1810.04805)
- A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman, ‘GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding’. arXiv, Feb. 22, 2019. Accessed: Jan. 25, 2024. [Online]. Available: [http://arxiv.org/abs/1804.07461](http://arxiv.org/abs/1804.07461)
- P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, ‘SQuAD: 100,000+ Questions for Machine Comprehension of Text’, in _Proceedings of the 2016 Conference on Empirical Methods in Natural          Language Processing_, Austin, Texas: Association for Computational Linguistics, 2016, pp. 2383–2392. doi: [10.18653/v1/D16-1264](https://doi.org/10.18653/v1/D16-1264).
