<h2 align="center"><b><h3>Evaluating Adaptive Activation Functions in Language Modelsn</h3></b></h2>

<p align="center">
  <b>Yijun Wu</b>
  <b> Supervisor:  Aral de Moor, Responsible Professors: Maliheh Izadi, Arie van Deursen </b>
</p>


<p align="center">
  <i>
    Delft University of Technology<br>
    CSE3000 Research Project EEMCS, Delft University of Technology, The Netherlands<br>
  </i>
</p>
<br>

<p align="center">
  <a href="https://www.overleaf.com/read/qqcrmwknqgvt#88bba0">
</p>

<br>

---
<br>
<h3 align="center"><b>Abstract</b></h3><br>
Although transformers are state-of-the-art models for natural language tasks, obtaining reasonable performance still often requires large transformers which are expensive to train and deploy. Fortunately, there are techniques to increase the size of transformers without extra computing costs. One such technique is sparsity. However, it remains unclear whether sparse architecture is intrinsically more efficient than its dense counterpart. In this paper, we investigate whether replacing the feedforward networks in small transformers with sparse alternatives results in better predictions and faster inference. We found that although inference speed does not increase due to software and hardware limitations, certain sparse alternatives do result in better language understanding. Our research contributes to smarter architectural decision making when designing small language models.
</br>

---
<br>

### Contents

This repository contains the (online) appendix, source code used to train our models, and the offline evaluation. 
This repository contains replication code for the paper "Sparse Transformers are (in)Efficient Learners: Comparing Sparse Feedforward Layers in Small Transformers".

## Installation
Make sure you have cuda installed. This code was tested with cuda 12.1 and 11.8

```bash
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Note: install torch for cuda version on your machine

## Dataset
The dataset should be placed in `tiny-transformers/data/TokenizedTinyStories` folder. 

## Replication

1. Go to `tiny-transformers/code/eugene` and run `pretrain_all.sh` This takes 2 arguments, the GPU number and whether to pretain in debug mode (`True/False`) on a dummy dataset. The models are saved under `tiny-transformers/models/eugene`
2. Go to `tiny-transformers/models/` and run `add_tokenizer.sh` This adds the tokenizers needed by the evaluation pipeline.
3. Go to `tiny-transformers/code/evaluation-pipeline` and run `blimp_all.sh ../../models/eugene` to get the BliMP results.
4. Similarly, `delftblue_superglue_all.sh ../../models/eugene` gets the SuperGLUE results, but this only works on DelftBlue.
5. Run `tiny-transformers/code/eugene/speed_test.ipynb` for the inference speed test.

