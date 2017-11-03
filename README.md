# ReadMe

This repo contains the files that implemented the experiments and the data used in the paper 'A Comprehensive Evaluation 
of Deep Learning for Link Prediction in Realistic Biomedical Graphs' by Gamal Crichton, Yufan Guo, Sampo Pyysalo and Anna Korhonen.

The *experiment_batch.sh* file contains the steps to automate the experiments. The .py files are used in the batch script. The folders LINE, DeepWalk, node2vec etc. contains implementations of those algorithms obtained freely on the internet. The *data* folder contains the files for the graphs used. In cases where the files are split into numbered segments, combine them before doing the experiments. They were only split to get around Github's file size limit.
The files are generally well-documented and the parameters that each script accepts are in the files.

Feel free to open an issue if anything does not work.

**Dependencies**

+ Python 2.7
+ Tensorflow and tflearn
+ Networkx
+ SciKit-learn
+ GEM
+ Numpy
+ Pandas

## License
The code is provided under MIT license and the other materials under Creative Commons Attribution 4.0. 
