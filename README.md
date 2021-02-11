# Position-Aware-Tagging-for-ASTE

[EMNLP 2020] [Position-Aware Tagging for Aspect Sentiment Triplet Extraction (In EMNLP 2020)](https://arxiv.org/abs/2010.02609)

# Task Description
Aspect Sentiment Triplet Extraction (ASTE) is the task of extracting the triplets of target entities, their associated sentiment, and opinion spans explaining the reason for the sentiment. This task is **firstly** proposed by (Peng et al., 2020) in the paper publised in AAAI 2020, [Knowing What, How and Why: A Near Complete Solution for Aspect-based Sentiment Analysis (In AAAI 2020)](https://arxiv.org/pdf/1911.01616.pdf)

For Example:

Given the sentence:

**The screen is very large and crystal clear with amazing colors and resolution .**

The objective of the Aspect Sentiment Triplet Extraction (ASTE) task is to predict the triplets:

**[('screen', 'large', 'Positive'), ('screen', 'clear', 'Positive'), ('colors', 'amazing', 'Positive'), ('resolution', 'amazing', 'Positive')]**
 
where a triplet consists of (target, opinion, sentiment).

# Requirement

conda create -n JET python=3.7 anaconda

conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch



Pytorch 1.4

Python 3.7.3  

Transformers

Bert-as-service

# Running with GloVe
```
python jet_o.py  
```
By default, the model runs on 2014 laptop dataset with provided hyper-parameters (M=2) without BERT.
Change line 20-27 for different datasets.
```
python jet_t.py  
```
By default, the model runs on 2015 reataurant dataset with provided hyper-parameters (M=2) without BERT.
Change line 20-27 for different datasets.


# Running with BERT
Please install [bert-as-service](https://github.com/hanxiao/bert-as-service) before Start the BERT service:

```
bert-serving-start -pooling_layer -1 -model_dir uncased_L-12_H-768_A-12 -max_seq_len=NONE -num_worker=2 -port=8880 -pooling_strategy=NONE -cpu -show_tokens_to_client
```

Then, 
```
python jet_o.py  
```
Change line 27 in the current file to True to runs on 2014 laptop dataset with provided hyper-parameters (M=2) with BERT.
Change line 20-27 for different datasets.
```
python jet_t.py  
```
Change line 27 in the current file to True to runs on 2015 reataurant dataset with provided hyper-parameters (M=2) with BERT.
Change line 20-27 for different datasets.

# Task Lists
- [ ] The current framwork only support BATCH_SIZE=1, more work need to be done to support batch calculation.

# Related Repo
The code are created based on the [StatNLP framework](https://github.com/sutd-statnlp/statnlp-neural).

