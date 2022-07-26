---
layout: single
permalink: /
hidden: true
classes: wide
sidebar:
  - title: "Visual Foresight With a Local Dyanmics Model"
    image: 
    text: "Colin Kohler, Robert Platt Northeastern University" 
---
**Abstract:** Model-free policy learning has been shown to be capable of learning manipulation policies which can 
solve long-time horizon tasks using single-step manipulation primitives. However, training these 
policies is a time-consuming process requiring large amounts of data. We propose the Local Dynamics 
Model (LDM) which efficiently learns the state-transition function for these manipulation primitives. 
By combining the LDM with model-free policy learning, we can learn policies which can solve complex 
manipulation tasks using one-step lookahead planning. We show that the LDM is both more sample-efficient 
and outperforms other model architectures. When combined with planning, we can outperform other 
model-based and model-free policies on several challenging manipulation tasks in simulation. 

# Paper
Our work has been accepted to the 54th International Symposium on Robotics (ISRR 2022). Currently a preprint is
avaliable on [Arxiv](https://arxiv.org/pdf/2206.14802.pdf).

# Idea
We investigate the use of visual foresight for use in complex robotic manipulation tasks through the use of a Local
Dynamics Model (LDM).

{% include figure image_path="/assets/images/ldm_ex.png" %}

# Video

# Code
The code for the Local Dynamics Model detailed in this work can be found [here](https://github.com/ColinKohler/LocalDynamicsModel).

# Citation
```
@misc{https://doi.org/10.48550/arxiv.2206.14802,
  doi = {10.48550/ARXIV.2206.14802},
  url = {https://arxiv.org/abs/2206.14802},
  author = {Kohler, Colin and Platt, Robert},
  title = {Visual Foresight With a Local Dynamics Model},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```

# Contact
If you have any questions, please feel free to contact [Colin Kohler](https://colinkohler.github.io/webpage/) at kohler[dot]c[at]northeastern[dot]edu.
