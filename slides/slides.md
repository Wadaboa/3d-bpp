---
theme: seriph
title: '3D-BPP'
titleTemplate: '%s - Slides'
background: https://www.publicdomainpictures.net/pictures/200000/velka/plain-red-background.jpg
class: 'text-center'
highlighter: 'prism'
lineNumbers: true
colorSchema: 'light'
drawings:
  persist: false
---

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

# Problem definition

- We are given a set of $n$ rectangular-shaped _items_, each characterized by width $w_j$, depth $d_j$ and height $h_j$ $(j\in J=\{1,\dots,n\})$
- We are also given an unlimited number of identical 3D containers (_bins_) having width $W$, depth $D$ and height $H$
- The _three-dimensional bin-packing problem_ (3D-BPP) consists in orthogonally packing all the items into the minimum number of bins

## Assumptions

- Items may not be rotated
- All input data are positive integers satisfying $w_j\le W, d_j\le D, h_j\le H$

---

# Lower bounds

1. $L_0=\left\lceil\frac{\sum_{j=1}^n v_j}{B}\right\rceil$, where $v_j=w_j\times d_h\times h_j$
    - Continuous lower bound: measures the overall liquid volume
    - Worst-case performance ratio of $\frac{1}{8}$
    - Time complexity: $O(n)$
1. $L_1=\max\{L_1^{WH},L_1^{WD},L_1^{HD}\}$
    - Obtained by reduction to the one-dimensional case
    - Worst-case performance arbitrarily bad
    - Time complexity: $O(n)$
1. $L_2=\max\{L_2^{WH},L_2^{WD},L_2^{HD}\}$
    - Explicitly takes into account the three dimensions of the items and dominates $L_1$
    - Worst-case performance ratio of $\frac{2}{3}$
    - Time complexity: $O(n^2)$

---

# Dataset

$$
\begin{array} {|l|l|l|}\hline 
  \textbf{Characteristic} & \textbf{Distribution} & \textbf{Parameters}        \\ \hline
\text{Depth/width sratio } R_{DW}       & \text{Normal}                & (0.695,0.118)  \\
\text{Height/width ratio } R_{HW}      & \text{Lognormal}             & (-0.654,0.453) \\
\text{Repetition } F             & \text{Lognormal}             & (0.544,0.658)  \\
\text{Volume } V     & \text{Lognormal}             & (2.568,0.705) \\ \hline
\end{array}
$$

- $(\frac{V}{R_{HW} \times R_{DH}})^{\frac{1}{3}}$

---

# Superitems, layers

Cosa sono? Perchè sono utili a risolvere il problema?
Tipi di Superitems
<!--
Note presentatore
-->

---

# Procedure approach

Mega schema di Fala
<!--
Note presentatore
-->
---

# Baseline model

Constraints usate, objective

Perchè fa schifo?
<!--
Note presentatore
-->
---

# Maxrects

Come funziona?
Perchè è molto utile?
I suoi limiti
<!--
Note presentatore
-->
---

# Column Generation

Suddivisione Master Subproblem

Duali, ecc (vedi readme.md)

Schema formule Master, Subproblem
<!--
Note presentatore
-->
---

# Layer filtering

Metodi per identificare i layer migliori, densità

<!--
Note presentatore
-->
---

# Bin packing

Posizionamento layer nei bin

Compattamento oggetti inseriti

---
# Future improvements

Uso di rotazione, peso, supporto, machine learning tra i sottoproblemi di CG

---
# Demo

Link streamlit
