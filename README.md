10th Private [Post Processing]

# RAM RAMRAKHYA part
Congrats to the winners and Thanks to my teammates. (Here's a brief overview of things that matter in our solution

## Loss function
We sticked for a long time with the BCE and couldn’t improve results by changing it. We however managed to find weights that nicely fitted our post-processing policy.
loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([
0.9, 1, 1.5, 0.8, 0.8, 0.8, 0.96, 1.1, 1.1, 3,  1, 1.1, 2, 3, 3,   2, 1, 2, 1, 2, 0.9, 0.75, 0.9, 0.75, 0.75, 0.7, 1, 2.5, 1, 0.75]))
Multiple other experiments with custom loss functions were conducted, but BCE performed the best.

## Training stuff
1 epoch with frozen encoder + 2 epochs with with everything unfrozen
Linear lr scheduling, with a custom learning rate depending on the layer. The transformer has a lower one as it is pretrained, and the closer to the output, the larger the learning rate
AdamW with betas=(0.5, 0.999) and a no bias decay of 1
Batch size is 64 for the first epoch, and then the larger we can fit on our gpus with sometimes an accumulation step

## Models
Our solution is a Ensemble of 4 models, (3 BERT-Large + 1 BERT-Base)

We made most experiments using the bert-base-uncased architecture, and managed to build a strong pipeline about 1 week before the end of the competition. This enabled us to switch easily to bigger ones which in the end made the strength of our ensemble.

We build 4 different architectures on top of the transformer, and used two of them overall.

## Bert Base Uncased
Here is the idea about the custom arch one we picked for our bert-base approach :

Input is [CLS] title [Q] question [A] answer [SEP] and [0, … 0, 1, …., 1, 2, …, 2] for ids. We initialed special tokens with values of the [SEP] token. Custom ids were also initialized with the model values.
Custom head : take the [CLS] token of the last n=8 layers, apply a dense of size m=1024 and a tanh activation, then concatenate everything.
Embeddings for the categoryand host column (+ tanh). We concatenate them with the output of the custom pooler and obtain the logits.
Some text cleaning (latex, urls, spaces, backslashes) was also applied
This model is the only one that uses text cleaning and embeddings, it helps for diversity I guess.

## Bert Larges
They repose on the same architecture. We use two inputs :
[Q] title [SEP] question [SEP]for tokens and[0, … 0, 1, …., 1]` for ids
[A] title [SEP] answer[SEP]for tokens and[0, … 0, 1, …., 1]` for ids
[Q] and [A] start with the value of the [CLS] token
Again, custom pooling head. Values of nand mare below.
Depending on the column, we either predict the value with only the pooled [Q] token, only the *
[A] token or both. The policy chosen is the following :
self.mix = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 2] # 0 = both, 1 = q, 2 = a
Concerning the pooler, it differs a bit :
Bert large cased : n = 24, m = 128
Bert large uncased : n = 24, m = 128
Bert large uncased wwm : n = 8, m = 768
Post-processing
Detailed here :

https://www.kaggle.com/c/google-quest-challenge/discussion/129901

Results
Single model
Our best model is the BERT-Large Whole Word Masking one, trained using Weighted BCE Loss gave :

With Post-Processing:

Private LB : 0.41919
Public LB : 0.46428
CV: 0.454
Without post-processing :

Private LB : 0.38127
Public LB : 0.40737
CV: 0.415
Which is a +0.06 boost on public and +0.04 on private. As you can see our single models are not that strong.

Ensemble
Our best selected solution is a simple average of the 4 mentioned model.
Private LB : 0.42430
Public LB : 0.47259

# THEO part

Hi everybody, I'm also sharing our post-processing approach that gave us about 0.05 boost on public.

The idea is to discretize the predictions, in order to benefit from the ambiguity due to the fact that equal predictions have the same ranking. Target values are discrete and some columns take very few values so it really helps.

To do so, we applly k-Means and replace values with the associated centroid. The number of cluster is determined using out of fold (oof) data, by optimizing the spearmanr for each column.

Then we do a bit of smoothing not to overfit too much on oof data.

We do not apply clustering for every column though, only when it helps by > 0.001.

On test time, here is what it looks :

post_processed_preds = pred_test.copy()

for col in range(pred_test.shape[1]):  
    if n_clusts:
        kmeans = KMeans(n_clusters=n_clusts)
        kmeans.fit(np.concatenate([pred_oof, pred_test])[:, col].reshape(-1, 1))
        preds = kmeans.cluster_centers_[kmeans.predict(pred_test[:, col].reshape(-1, 1))].reshape(-1)
       post_processed_preds[:, col] = preds
Note that we fit the clustering on pred_oof and pred_test, it helps as well (+0.02 public).

question_type_spelling didn't like our post-processing too much and we lost quite a lot of subs.

Congratz to @cl2ev1 for figuring the idea, with some tweaks it did wonders. We'll cover other parts of our solution soon.