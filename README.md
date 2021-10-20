# Drop-DTW: Aligning Common Signal Between Sequences While Dropping Outliers

*[Nikita Dvornik](https://thoth.inrialpes.fr/people/mdvornik/)*<sup>1</sup>, 
*[Isma Hadji](http://www.cse.yorku.ca/~hadjisma/)*<sup>1</sup>, 
*[Konstantinos G. Derpanis](https://www.cs.ryerson.ca/kosta/)*<sup>1</sup>, 
*[Allan D. Jepson](https://www.cs.toronto.edu/~jepson/)*<sup>1</sup>,
and * [Animesh Garg] (https://animesh.garg.tech/)*<sup>1</sup>

<sup>1</sup>Samsung AI Center (SAIC) - Toronto &nbsp;&nbsp;
* This research was conducted at SAIC-Toronto, funded by Samsung Research, and a provisional patent application has been filed.


#
<div align="center">
  <img src="demo/teaser.png" width="600px"/>
</div>

In this work, we consider the problem of sequence-to-sequence alignment for signals
containing outliers. Assuming the absence of outliers, the standard Dynamic
Time Warping (DTW) algorithm efficiently computes the optimal alignment between
two (generally) variable-length sequences. While DTW is robust to temporal
shifts and dilations of the signal, it fails to align sequences in a meaningful way
in the presence of outliers that can be arbitrarily interspersed in the sequences. To
address this problem, we introduce Drop-DTW, a novel algorithm that aligns the
common signal between the sequences while automatically dropping the outlier elements
from the matching. The entire procedure is implemented as a single dynamic
program that is efficient and fully differentiable. In our experiments, we show that
Drop-DTW is a robust similarity measure for sequence retrieval and demonstrate
its effectiveness as a training loss on diverse applications. With Drop-DTW, we
address temporal step localization on instructional videos, representation learning
from noisy videos, and cross-modal representation learning for audio-visual
retrieval and localization. In all applications, we take a weakly- or unsupervised
approach and demonstrate state-of-the-art results under these settings.

## Applications
The proposed alignment loss enables various downstream applications. Take a look at this video for examples.
[![Watch the video](demo/supp.png)](https://youtu.be/)

## Code
Code for this project will be availabe here soon! stay tuned...

## Citation
If you use this code or our models, please cite our paper:
```
@inproceedings{Drop-DTW,
  title={Drop-DTW: Aligning Common Signal Between Sequences While Dropping Outliers},
  author={Dvornik, Nikita and Hadji, Isma and Derpanis, Konstantinos G and Garg, Animesh and Jepson, Allan D},
  booktitle={NeurIPS},
  year={2021}
}
```
