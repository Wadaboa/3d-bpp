---
theme: seriph
title: '3D-BPP'
titleTemplate: '%s - Slides'
background: https://www.publicdomainpictures.net/pictures/200000/velka/plain-red-background.jpg
class: 'text-center'
highlighter: 'prism'
lineNumbers: false
colorSchema: 'light'
drawings:
  persist: false
download: true
---

<style>
  p {
    opacity: 1.0 !important;
  }
</style>

# 3D Bin Packing

Artificial Intelligence in Industry - Final Project <br> 
University of Bologna - Academic Year 2020/21

<div class="pt-12">
  Leonardo Calbi, Lorenzo Cellini, Alessio Falai
</div>

<div class="abs-br m-6 flex gap-2">
  <a href="https://github.com/Wadaboa/3d-bpp" target="_blank" alt="GitHub"
    class="text-xl icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

---

# 3D-BPP

## Problem definition

- We are given a set of $n$ rectangular-shaped _items_, each characterized by width $w_j$, depth $d_j$ and height $h_j$
- We are given a number of identical 3D containers (_bins_) having width $W$, depth $D$ and height $H$
- The _three-dimensional bin-packing problem_ (3D-BPP) consists in orthogonally packing all the items into the minimum number of bins

## Assumptions

- Items may not be rotated
- We have unlimited bins $b_1,b_2,\dots$ at our disposal
- All input data are positive integers satisfying $w_j\le W, d_j\le D, h_j\le H$

---

# Lower bounds

1. $L_0=\left\lceil\frac{\sum_{j=1}^n v_j}{B}\right\rceil$, where $v_j=w_j\times d_h\times h_j$
    - Continuous lower bound: measures the overall _liquid volume_
    - Worst-case performance ratio of $\frac{1}{8}$
    - Time complexity: $O(n)$
1. $L_1=\max\{L_1^{WH},L_1^{WD},L_1^{HD}\}$
    - Obtained by reduction to the _one-dimensional_ case
    - Worst-case performance arbitrarily bad
    - Time complexity: $O(n)$
1. $L_2=\max\{L_2^{WH},L_2^{WD},L_2^{HD}\}$
    - Explicitly takes into account the _three dimensions_ of the items and dominates $L_1$
    - Worst-case performance ratio of $\frac{2}{3}$
    - Time complexity: $O(n^2)$

---

# Dataset

## Distributions
$$
\begin{array} {|l|l|l|}\hline 
  \textbf{Characteristic} & \textbf{Distribution} & \textbf{Parameters}        \\ \hline
\text{Depth/width ratio } R_{DW}       & \text{Normal}                & (0.695,0.118)  \\
\text{Height/width ratio } R_{HW}      & \text{Lognormal}             & (-0.654,0.453) \\
\text{Repetition } F             & \text{Lognormal}             & (0.544,0.658)  \\
\text{Volume } V     & \text{Lognormal}             & (2.568,0.705) \\ 
\text{Weight } L     & \text{Lognormal}             & (2,2) \\ \hline
\end{array}
$$

## Reasoning
1. Volumes: $V\sim LN (\mu, \sigma^2), \mu=\frac{\sum_{j=1}^{N}\log v_j}{N}, \sigma^2=\frac{\sum_{j=1}^{N}(\log v_j-\mu)^2}{N},N=166407, j\in\{C_1,\dots C_5\}$
1. Widths: $W=(\frac{V}{R_{DW} \times R_{HW}})^{\frac{1}{3}}$
1. Depths: $D=W\times R_{DW}$
1. Heights: $H=W\times R_{HW}$

---

# Superitems

<p align="center">
  <img src='/superitems.png' width='700'/>
</p>

- A _superitem_ is a collection of individual items that are compactly stacked together
- Building procedure
    1. _Single_: composed of a unique item
    2. _Horizontal_: composed of items having the exact same dimensions
        - Possibility to restrict their generation (_2D_, _2W_, _4_ or none) 
    3. _Vertical_: composed of items s.t. the ones on top have an area support of at least $70\%$
        - Maximum $M$ stacked superitems (either single or horizontal)

---

# Layers

<p align="center">
  <img src='/2d-layers.png' width='300'/>
</p>

- A _layer_ is defined as a two-dimensional arrangement of items within the horizontal boundaries of a bin with no superitems stacked on top of each other
- Superitems are placed relative to layers and layers are placed relative to bins

---

# Workflow

<p align="center">
  <img src='/flow.png' width='320'/>
</p>

---
layout: two-cols
---

# Baseline

<p align="left">
  <img src='/sppsi.png' width="400" style='margin-top: 60px'/>
</p>

::right::

# Constraints

- (5): ensure that every item is included in a layer
- (6): define the height of layer $l$
- (7): redundant valid cuts that force the area of a layer to fit within the area of a bin
- (8): enforce at least one relative positioning relationship between each pair of items in a layer
- (9) and (10): ensure that there is at most one spatial relationship between items $i$ and $j$ along each of the width and depth dimensions 
- (11) and (12): non-overlapping constraints
- (13) and (14): ensure that items are placed within the boundaries of the bin

---
layout: two-cols
---

# `MAXRECTS`

<p align="center">
  <img src='/maxrects.png' width="380" style='margin-top: 50px'/>
</p>

::right::

# Details

- `MAXRECTS` is a _procedural_ algorithm for solving the 2D bin packing problem, based on an extension of the `GUILLOTINE` split rule
- _Height groups_: divide the whole pool of superitems into groups having heights within a given tolerance
- `MAXRECTS` is used to generate layers
- Run _multiple strategies_ (Bottom-Left, Best Area Fit, Best Short Side Fit and Best Long Side Fit) and select the most dense layers

---

# Column generation

<p align="center">
  <img src='/cg.png' width="250" />
</p>

- _Warm start_: single item layers vs `MAXRECTS`
- Each iteration builds only a _single layer_ and adds it to the whole pool
- _Stopping criterion_: maximum iterations or convergence (non-negative reduced costs)
- _Optimality_: no branch-and-price scheme 

---

# `RMP`

<p align="center">
  <img src='/rmp.png' width="600"/>
</p>

- `RMP` selects the best layers so far
- $\alpha_l\ge 0$ represents the linear relaxation of the integrality constraint $\alpha_l\in\{0,1\}$
- $\lambda$ are the dual variables corresponding to constraints (18)
- The master problem is solved using `BOP` (it only contains boolean variables), while the reduced one is solved with `GLOP` (linear program)

---

# `SP`

<p align="center">
  <img src='/sp.png' width="600"/>
</p>

- `SP` selects items and positions them in a new layer
- $o^l-\sum_i\sum_s\lambda_i f_{si} z_{sl}$ is the reduced cost of a new layer $l$
- `SP` can be solved in the following ways
    - `MAXRECTS`: solve the whole pricing subproblem heuristically, using `MAXRECTS` to place superitems by biggest duals first
    - _Placement_ and _no-placement_ strategy
        1. _No-placement_: serves as an item selection mechanism, thus ignoring the expensive placement constraints (MIP or CP)
        1. _Placement_: checks whether there is a feasible placement of the selected items in a layer and places them if possible, otherwise iterates with the no-placement model (MIP or CP or `MAXRECTS`)

---

# Layer filtering

1. _Discard layers by densities_ (given a minimum density $d_m$)
1. _Discard layers by item coverage_
      - If at least one item in layer $l$ was already selected more times than $M_a$, discard $l$
      - If at least $M_s$ items in layer $l$ are already covered by previously selected layers, discard $l$
2. _Remove duplicated items_
      - Remove superitems in different layers containing the same item (remove the ones in less dense layers)
      - Remove superitems in the same layer containing the same item (remove the ones with less volume)
      - Re-arrange layers (using `MAXRECTS`) in which at least one superitem was removed (if $d_l > d_m$)
3. _Remove empty layers_
4. _Sort layers by densities_


---

# Bin packing

<p align="center">
  <img src='/bin-example.png' width="300"/>
</p>

1. _Uncovered items_: create new layers filled with items that were not covered in the previous procedures and add them to the layer pool
1. _Bins building procedure_: stack layers on top of each other until a bin is full and open new bins as needed
1. _Compact bins_: let items "fall" to the ground as much as possible, without allowing intersections


---

# Future improvements

<p align="center">
  <img src='/container-example.png' width="350"/>
</p>

- Allow items to be _rotated_ on the 3 axis
- Integrate the _branch-and-price_ scheme into column generation to prove optimality
- Handle _weight_ constraints and bin load capacity
- Improve item _support_ through MP models (as described in the paper)
- Load bins inside _containers_

---
layout: center
---

# Demo

```python
python3 -m streamlit run src/dashboard.py
jupyter notebook bpp.ipynb
```

---
layout: end
---

Bye bye
