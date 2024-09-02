# Speech dereverberation constrained on room impulse response characteristics

[Paper](https://www.doi.org/10.21437/Interspeech.2024-1173) &nbsp;
<a href="https://www.arxiv.org/abs/2407.08657">Arxiv</a> &nbsp;
<a href="https://hal.science/hal-04640068/">HAL</a> &nbsp;
<a href="https://www.github.com/Louis-Bahrman/SD-cRIRc">Code</a> &nbsp;
[Poster](docs/poster.pdf)

![Block Diagram](docs/block_diagram.svg "Block Diagram")

## Abstract

Single-channel speech dereverberation aims at extracting a dry speech signal from a recording affected by the acoustic reflections in a room. However, most current deep learning-based approaches for speech dereverberation are not interpretable for room acoustics, and can be considered as black-box systems in that regard. In this work, we address this problem by regularizing the training loss using a novel physical coherence loss which encourages the room impulse response (RIR) induced by the dereverberated output of the model to match the acoustic properties of the room in which the signal was recorded. Our investigation demonstrates the preservation of the original dereverberated signal alongside the provision of a more physically coherent RIR.

## Model Weights

Model Weights can be obtained from  `louis [ɗօt] bahrman [аt] telecom-paris.fr`

## Citing this work

If you use this work in your research or business, please cite it using the following BibTeX entry:

```
@inproceedings{bahrman24_interspeech,
  title     = {Speech dereverberation constrained on room impulse response characteristics},
  author    = {Louis Bahrman and Mathieu Fontaine and Jonathan {Le Roux} and Gaël Richard},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {622--626},
  doi       = {10.21437/Interspeech.2024-1173},
}
```
