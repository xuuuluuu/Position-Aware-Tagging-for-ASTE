# Position-Aware-Tagging-for-ASTE

[EMNLP 2021] [Position-Aware Tagging for Aspect Sentiment Triplet Extraction (In EMNLP 2021)](https://github.com/xuuuluuu/Position-Aware-Tagging-for-ASTE)

# Requirement
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

