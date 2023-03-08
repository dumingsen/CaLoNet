# Causal and Local Correlations Based Network for Multivarite Time Series Classification

**The repository is for a paper on this topic under review.**

Recently, time series classification has attracted the attention of a large number of researchers. And hundreds of methods have been proposed. However, these methods often ignore the spatio correlations among dimensions and the local correlations among features.  For this, the Causal and Local correlations based Network (CaLoNet) is proposed for multivarite time series classification in this study. Firstly, pairwise spatio correlations between dimensions using causality modeling and obtains the graph structure. And then, a relationship extraction network is used to fuse local correlations to obtain long-term dependency features. Finally, the graph structure and long-term dependency features are integrated into the graph neural network. Experiments on the UEA datasets show that CaLoNet can  obtain a competitive performance with the state-of-the-art methods.

## Requirements
* Python 
* PyTorch 

## Datasets
Get MTS datasets in http://timeseriesclassification.com/dataset.php.

**Train/Test**:
```bash
run: python run_UEA.py
```

## Acknowledgements
This work was supported by the Innovation Methods Work Special Project under Grant 2020IM020100, and the Natural Science Foundation of Shandong Province under Grant ZR2020QF112.

We would like to thank Eamonn Keogh and his team, Tony Bagnall and his team for the UEA/UCR time series classification repository.

