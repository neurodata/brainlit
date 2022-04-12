---
marp: true
theme: poster
paginate: false
size: 36:24
---
<div class="header">
<div>

<!-- <div class=center_container> -->

![headerlogo](../images/hopkins-logo.png)


</div>
<div>

# Graphical User Interface for Semi-Automated Tracing of Neuronal Processes


## Thomas L. Athey<span class=super>1*</span>, Michael I. Miller<span class=super>1</span>

##### 1 - Department of Biomedical Engineering, Johns Hopkins University  <br>$\ast$ - correspondence: ![icon](../images/email.png) [_tathey1@jhu.edu_](mailto:tathey1@jhu.edu) ![icon](../images/github.png) [_@tathey1 (Github)_](https://github.com/tathey1) ![icon](../images/twitter.png) [_@Thomas_L_Athey (Twitter)_](https://twitter.com/Thomas_L_Athey)

</div>
<div>

![headerlogo](../images/nd_logo.png)

<span style="text-align:center; margin:0; padding:0">

</span>

</div>
</div>

<span class='h3-noline'> Summary </span>


<div class='box'>
<div class="columns5">
<div>


- Aimed to define bilateral symmetry for a connectome, and formally test this hypothesis.

</div>
<div>

- Hemispheres differ in a network-wide parameter under even the simplest model of a network pair.

</div>
<div>

- Hemispheres differ in neuron group connection probabilities, even when adjusting for the network-wide effect.

</div>
<div>

- Detect no differences in adjusted group connections after removing a cell type or when only considering strong edges.

<!-- - Removing a specific cell type and adjusting for this network-wide effect provides one notion of bilateral symmetry -->

<!-- 
- Difference between hemispheres can be explained as combination of network-wide and cell type-specific effects -->

</div>
<div>

- Provided a definition of bilateral symmetry exhibited by this connectome, tools for future connectome comparisons

</div>
</div>
</div>

<div class="columns3">
<div>


### Motivation

- A neuron's morphology determines how it integrates into brain circuits and contributes to overall brain function.
- Efforts to establish brain-wide atlases of neuron morphology in the mouse rely on laboroious manual tracing [1].
- Future work in human brains will exacerbate this bottleneck.

### Data


![](../images/ng.png)


**Fig 1:** Sample from Janelia Mouselight project. Sparse labeling is achieved using a combination of diluted AAV Syn-iCre and a Cre-dependent reporter. Images are acquired by serial two-photon tomography at $0.3 \times 0.3 \times 1.0 \mu m^3$  resolution.
</div>
<div>


### Group connection test (Model 2)

![center w:10.5in](./../../../results/figs/sbm_unmatched_test/sbm_methods_explain.svg)
**Fig 3A:** Testing under stochastic block model (SBM) compares probabilities of connections between groups (here using cell types [1]).

<!-- START subcolumns -->
<div class=columns2>
<div>

![](../../../results/figs/sbm_unmatched_test/sbm_uncorrected_pvalues.svg)

</div>
<div>

![center w:5in](../../../results/figs/sbm_unmatched_test/significant_p_comparison.svg)

</div>
</div>

<div class=columns2>
<div>


**Fig 3B:** Test comparing group connections rejected ($p{<}10^{-7}$); five specific connections differ.

</div>
<div>

**Fig 3C:** For significant group connections, denser hemisphere probability is always higher.

</div>
</div>

### Density-adjusted group connection test (Model 3)

<div class=columns2>
<div>

<br>

![center w:5in](./../../../results/figs/adjusted_sbm_unmatched_test/adjusted_methods_explain.svg)

</div>
<div>

![](./../../../results/figs/adjusted_sbm_unmatched_test/sbm_pvalues.svg)

</div>
</div>

<div class=columns2>
<div>

**Fig 4A:** Hypothesis from Fig 3 modified by a factor $c$ set to make densities equal.

</div>
<div>

**Fig 4B:** Test comparing adjusted group connections rejected $(p{<}10^{-2})$; differences from KCs.

</div>
</div>


</div>
<div>

### Notions of bilateral symmetry


<div class="columns2">
<div>

#### With Kenyon cells
| Model |                       $H_0$ (vs. $H_A \neq$)                       |    p-value    |
| :---: | :----------------------------------------------------------------: | :-----------: |
| **1** |  $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  | ${<}10^{-23}$ |
| **2** | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  | ${<}10^{-7}$  |
| **3** | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ | ${<}10^{-2}$  |


</div>
<div>

#### Without Kenyon cells
| Model |                       $H_0$ (vs. $H_A \neq$)                       |    p-value    |
| :---: | :----------------------------------------------------------------: | :-----------: |
| **1** |  $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  | ${<}10^{-26}$ |
| **2** | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  | ${<}10^{-2}$  |
| **3** | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ |    $0.51$     |

</div>
</div>


### Edge weight thresholds



### Limitations and extensions
- Other models to consider (e.g. random dot product graph [3])
- Other sensible neuron groupings for group connection test
- Matching nodes across networks leads to new models, likely more power

###


<div class="columns2">
<div>

#### Code


#### Acknowledgements
<footer>
We thank the MouseLight team at HHMI Janelia for providing us with access to this data, and answering our questions about it. Thanks to Benjamin Pedigo for providing the template for this poster.
</footer>

</div>
<div>

#### References

<footer>
[1] Winnubst
<br>
</footer>

#### Funding

<footer>
Funding
</footer>

</div>
</div>


</div>
</div>
