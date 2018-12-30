## NOTICE
#### As the volume of dataset is too large to be uploaded, please put the dataset in src/trainer.
#### If you don't have bin file, you can convert txt file to bin file using the following python code.
#### Also, you can download glove data from https://nlp.stanford.edu/projects/glove/ .

```
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec("trainer\glove.6B.100d.txt","trainer\glove.6B.100d.bin")
```
