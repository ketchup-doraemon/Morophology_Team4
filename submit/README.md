
## Team 4

GloVe data was downloaded from https://nlp.stanford.edu/projects/glove/ .  
All Glove data is expected to convert its format using the following code.  
```
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec("trainer\glove.6B.100d.txt","trainer\glove.6B.100d.bin")
```

Also, the following modules are expected to be installed.  
chainer 5.1.0     
gensim  3.6.0    

## Model 1
model1.ipynb : Word Embedding Space Model.  
sampling_from_model1.ipynb : Sampling the pseudo teacher data for model 4.  


## Model 2
model2_premodeling.ipynb : A pre-Test of Charcter Prediction Model.  
model2_with_allwords.ipynb : Charcter Prediction Model without GloVe Vector.  
model2_with_glove.ipynb : Charcter Prediction Model with GloVe Vector. This is final version of model 2.  

## Model 3
model3.ipynb : Three Net Model.  

## Model 4
model4_attention.ipynb : An original Attention Model.  
model4_train.ipynb : Augmented Attention Model with GloVe data. This is main code of model4.  
model4_without_glove.ipynb : Augmented Attention Model without GloVe data.  
model4_comparison.ipynb : Comparison with model 1.  

