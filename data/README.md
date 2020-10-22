The files in the **triplet_data** folder are pre-processed from **ASTE-Data-V2** folder to fulfill the requirements of our model.



The files in the **ASTE-Data-V2** folder are originally released in this [site](https://github.com/xuuuluuu/SemEval-Triplet-data).

The data has the following format: 

> sentence####[(target position, opinion position, sentiment)]

If there are multiple triplets in the same sentence:

> sentence####[(target position, opinion position, sentiment), ..., (target position, opinion position, sentiment)]

For example:

> The screen is very large and crystal clear with amazing colors and resolution .####[([1], [4], 'POS'), ([1], [7], 'POS'), ([10], [9], 'POS'), ([12], [9], 'POS')]
