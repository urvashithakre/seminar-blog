---
permalink: /
title: "ProtGPT2 - A deep unsupervised language model for protein design"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
<style>
p {
    text-align: justify;
}
</style>
## 1. Introduction
 Proteins are essential molecules responsible for virtually all functions in living organisms. Designing new proteins could lead to breakthroughs in drug development, material science, and synthetic biology. However, this process is traditionally slow, resource-intensive, and highly specialized.

Recent progress in Transformer-based architectures has enabled the implementation of language models capable of generating text with human-like capabilities. Motivated by this success, researchers created ProtGPT2, a language model that was trained on the protein space and produces de novo protein sequences based on natural ones.

In this blog, we will explore how researchers are rethinking protein design using tools from an entirely different domain — natural language processing. We’ll look at how a language model was trained to generate protein sequences, how well those sequences mimic natural proteins and what this could mean for the future of biology.

Whether you’re coming from biology, computer science, or just curious about AI, this post will walk you through the fascinating crossroad where deep learning meets molecular design.

## 2. From Language to Biology: A Shared Structure
Recent breakthroughs in Natural Language Processing (NLP) have demonstrated that large language models (LLMs) can effectively learn the structure, meaning, and composition of human language. These models are trained on vast amounts of text, enabling them to generate coherent paragraphs, translate across languages and even write poetry.

Interestingly, proteins share a surprisingly similar structure to natural language. Proteins are linear chains of amino acids drawn from a 20-character chemical alphabet. Like natural languages, protein sequences are information-complete, storing structure and functioning completely in the order of their amino acids with remarkable efficiency. Amino acids form structured domains that fold into functional proteins, much like words use grammar to form sentences. As a result of this analogy, protein sequences are now thought of as a type of language in which structure and function are determined by sequence.

## 3. Previous Work
Before ProtGPT2, several models laid the groundwork for applying NLP techniques to biological sequences. These studies demonstrated that proteins when treated like language can be analyzed and even generated using the same tools that revolutionized text processing.

The three main phases of protein language modelling are: supervised learning, unsupervised learning, and autoregressive generation. Each of these contributed unique insights into how AI can understand and create biologically relevant sequences.

### 3.1 Supervised Models
Many earlier models were trained on labeled data, focusing on specific prediction tasks such as:
- Secondary structure prediction
- Stability assessment
- Homology detection

Platforms like BioSeq-BLM collected numerous supervised language models designed for biomolecular tasks. However, supervised learning has limitations: it requires curated datasets and is narrowly focused on predefined task, offering little flexibility for generative tasks.

### 3.2 Unsupervised Models
The rise of Transformer architectures introduced a shift toward unsupervised learning, where models learn from raw sequences without labels. Notable models include:

| Model         | Architecture  | Focus                        |
| ------------- | ------------- | ---------------------------- |
| **ProtBERT**  | BERT-style    | Embedding proteins           |
| **TCR-BERT**  | BERT-style    | T-cell receptor modeling     |
| **ProtTrans** | BERT & T5 mix | Multi-task protein NLP       |
| **ESM**       | Transformer   | Large-scale protein modeling |

Typically, masked language modelling was used to train these models, in which specific tokens are concealed and the model is trained to reconstruct them. They were not optimised for generation, but they worked well for embedding sequences.

### 3.3 Autoregressive Models for Protein Generation
To move from understanding to generation, researchers turned to autoregressive models - a class of models that predict each token based on the sequence of previous tokens. This is the core mechanism behind models like GPT.

Key autoregressive protein models prior to ProtGPT2 include:

- ProGen: One of the first autoregressive models to generate proteins
- RITA: A family of generative Transformer models
- DARK: Focused on de novo protein generation

The foundation for ProtGPT2 was established by these models, which advanced the field from static knowledge to dynamic, generative modelling of protein space.

## 4. From Foundations to Frontier: Meet ProtGPT2
The journey from treating protein sequences as linguistic constructs to generating new, functional proteins culminates in ProtGPT2—a large-scale autoregressive language model tailored specifically for protein design. While previous models focused on embeddings or prediction, ProtGPT2 was explicitly trained to generate realistic, diverse, and foldable protein sequences. It does so by learning the grammar and structure of proteins much like GPT-2 learns human language.

Let's dive deeper into how ProtGPT2 was built—from its architecture to its training dataset and tokenization strategy.
### 4.1 Model Architecture
ProtGPT2 is a big step forward in using deep learning to design proteins, building on recent progress in language modelling. Based on the GPT-2 architecture, ProtGPT2 is an autoregressive Transformer model with 738 million parameters. That means it generates outputs sequentially, one token at a time, conditioned only on what came before - perfect for modeling protein sequences. 

Formally, given a protein sequence: 
$$ W = \{ w_1, w_2, \dots, w_n \} $$

the model learns to predict the probability of the sequence as:
$$
p(\mathcal{W}) = \prod_{i=1}^{n} p(w_i \mid w_{<i})
$$

The training process minimizes the **negative log-likelihood** over all protein sequences in the dataset:

$$
\mathcal{L}_{\text{CLM}} = - \sum_{k=1}^{|D|} \sum_{i=1}^{|w_k|} \log\, p_\theta(w_{k,i} \mid w_{k,<i})
$$

Where:
- $$w_{k,i}$$: *i*-th amino acid in the *k*-th protein sequence  
- $$D$$: Protein dataset (UniRef50)  
- $$\theta$$: Model parameters  
- $$\mathcal{L}_{\text{CLM}}$$: Causal Language Modeling loss

This architecture enables the model to learn complex statistical dependencies — such as conserved motifs and structural sub-patterns — directly from sequence data.

### 4.2 The Dataset
ProtGPT2 was trained on UniRef50 (version 2021_04) — a clustered subset of UniProt that reduces redundancy by grouping sequences with >50% identity. This ensures both diversity and robustness during training.

| Subset       | Sequences           |
| ------------ | ------------------- |
| Training Set | \~44.9 million      |
| Validation   | \~4.9 million (10%) |

This dataset spans both known and “dark” proteome regions — proteins without known structure or function, enabling ProtGPT2 to generalize across structured and unexplored sequence space.

### 4.3 Byte Pair Encoding (BPE) for Tokenization
ProtGPT2 employs a Byte Pair Encoding (BPE) tokeniser, in contrast to conventional models that handle each amino acid as a distinct token. This enables the model to compress frequently occurring motifs and sub-sequences into higher-level tokens. ProtGPT2’s tokenizer was trained on the Swiss-Prot subset, a high-quality, manually curated collection from UniProt. This ensures robust vocabulary learning while minimizing noise from low-confidence sequences.

Here’s a quick breakdown of the tokenizer setup:
- Vocabulary size: 50,256 tokens
- Average token: ~4 amino acids
- Trained on: Swiss-Prot subset for robustness

This strategy reduces sequence length, improves generalization and helps the model learn biologically meaningful patterns.

### 4.4 Final Model Configuration

| Component    | Description                       |
|------------- |-----------------------------------|
| Architecture | GPT-2 large (decoder-only)        |
| Layers       | 36                                |
| Parameters   | 738 million                       |
| Batch Size   | 65,536 tokens per batch           |
| Optimizer    | Adam (β₁ = 0.9, β₂ = 0.999)       |
| Hardware     | 128 NVIDIA A100 GPUs for 4 days   |

<div style="text-align: center;">Model configuration</div>

Unlike masked models focused on classification or embedding, ProtGPT2 was explicitly trained for sequence generation, enabling it to compose entirely new proteins that closely resemble natural ones. To summarize, ProtGPT2 combines a powerful GPT-2 architecture with a massive protein sequence corpus (UniRef50) and a subword-aware BPE tokenizer. These elements work together to give the model the ability to understand the underlying "language" of proteins and produce new sequences that accurately represent their structural and functional characteristics.

![Fig. 1: ProtGPT2 Architecture](images/ProtGPT_Architecture.png)
<div style="text-align: center;">Figure 1: ProtGPT2 Architecture </div>

## 5. Decoding Strategies: How ProtGPT2 Generates Sequences?
Once ProtGPT2 is trained to model the protein language, the next step is generating new sequences. But how exactly are these sequences "sampled" from the model?

After our model has been trained, we must choose how to use it to produce sequences. The quality of the output is significantly impacted by how we sample from the probability distribution over potential amino acids that the model provides at each step.

### 5.1 Sampling Strategies Compared
Here are the primary decoding strategies explored:

| Strategy           | Description                                                           | Outcome                              |
|--------------------|-----------------------------------------------------------------------|--------------------------------------|
| **Greedy**         | Always selects the most probable amino acid at each step              | Repetitive, low-diversity sequences  |
| **Beam Search**    | Maintains multiple candidate sequences and picks the best-scoring one | Slightly better but still repetitive |
| **Random (Top-k)** | Samples from top-k probable tokens randomly                           | Diverse and biologically realistic   |

<div style="text-align: center;">Comparison of Greedy, Beam, and Top‑k sampling strategies</div>

Here's a visual representation of how these 3 strategies work.

<center>
  <img src="images/Sampling_Strategies.png" alt="Sampling Strategies" width="80%">
</center>

### 5.2 Which Strategy Works Best?
The authors found that Top-k sampling (k = 950) combined with a repetition penalty of 1.2 yielded the most realistic and diverse protein sequences.

This approach strikes a balance between:
- Structure: preserving plausible secondary and tertiary motifs
- Diversity: enabling the generation of novel and unique sequences

In contrast, Greedy and Beam Search decoding produced highly repetitive outputs which is undesirable for protein design, where structural and functional variety is important.

Here’s a visual representation of how decoding strategies affect output:

![](images/Sampling Output.jpg)
<div style="text-align: center;">Figure 2:Sampling outputs for GPT2-like language models on both text (a–d) and protein sequences (e–h). Repetitive sequences are generated by greedy and beam search; natural-like diversity emerges with random top-k sampling (g, h).</div>


## 6. Evaluating the Biological Plausibility of ProtGPT2 Sequences
After training, it’s critical to assess whether ProtGPT2 actually generates plausible, stable and structured proteins - not just random chains of amino acids.

To this end, researchers evaluated ProtGPT2 outputs across three biological axes:
1. Globularity & Order
2. Secondary Structure Composition
3. Similarity to Natural Proteins

### 6.1 Globularity & Disorder
Proteins come in various structural forms — some are flexible and disordered, others fold tightly into compact, globular forms. Globular proteins are the workhorses of biology, typically performing essential functions within cells. To assess ProtGPT2’s biological realism, the authors examined whether its generated sequences resemble globular proteins, much like natural ones.

Using IUPred3 — a tool that predicts whether a protein region is ordered or disordered — they analyzed 10,000 ProtGPT2-generated sequences and compared them with 10,000 natural proteins.

| Property                       | Natural Proteins | ProtGPT2 Sequences |
| ------------------------------ | ---------------- | ------------------ |
| **Globular domains (IUPred3)** | 88.40%           | 87.59%             |
| **Ordered amino acids**        | 82.59%           | 79.71%             |

These results are strikingly close. Despite being generated from scratch, ProtGPT2 sequences mimic the order and globularity found in real-world proteins — even without supervision or explicit structural constraints.

### 6.2 Secondary Structure Composition
Protein function heavily depends on secondary structure elements like alpha-helices and beta-sheets. So the team used PSIPRED, a well-known structure predictor, to further evaluate how ProtGPT2’s sequences stack up. 
 
![](images/Secondary_Structure_Comparison.png)
<div style="text-align: center;">Figure 3: Secondary structure comparison </div>

Again, the similarities are remarkable — ProtGPT2 is not only generating coherent protein sequences but ones with realistic structural patterns. Even without explicitly being trained on structure, ProtGPT2 captures the patterns that govern natural protein folding. That’s the power of deep learning on biological language.

### 6.3 Sequence Similarity & Novelty
One might wonder: are these sequences just copying known proteins, or are they genuinely novel? To answer this, the researchers turned to HHblits, a tool for detecting remote protein homology using profile hidden Markov models. In bioinformatics, two sequences are considered evolutionarily related (homologous) if they are similar above a certain threshold.Th e HSSP curve defines this threshold: it sets the minimum percent identity required, depending on the length of the alignment.


They performed a comparison using three datasets:

1. Natural proteins (yellow)
2. ProtGPT2-generated sequences (green)
3. Completely random sequences (red)

Each sequence was searched against the Uniclust30 database to find the best-matching natural counterpart. 

![](images/Sequence_Identities.jpg)
<div style="text-align: center;">Figure 4: Pairwise sequence identities vs. alignment length for each of the datasets </div>


**Takeaways from the Plot**
- Panel a (Natural Proteins): ~96% of sequences show strong similarity to real proteins.
- Panel b (ProtGPT2): ~93% of sequences are remotely but significantly similar to known proteins.
- Panel c (Random): Only ~7% show any resemblance — indicating ProtGPT2 isn’t just generating gibberish.

This striking result shows that ProtGPT2 isn't copying, but learning the "grammar" of protein sequences. It creates new, meaningful variations that resemble natural proteins, yet are not directly present in the training set.

**Interpretation:**

- The majority of ProtGPT2 sequences exhibit remote but biologically meaningful similarity to natural proteins.
- However, unlike direct memorization, these matches are often shorter and less identical, suggesting that ProtGPT2 creates new variations, not just regurgitations.
- Random sequences fail miserably — only 7% have meaningful hits.

Bottom Line: ProtGPT2 sequences are “similar but novel.” They are grounded in the statistical patterns of nature but diverge just enough to be considered creative.

## 7. Structural and Functional Validation of ProtGPT2 Sequences

Designing new proteins isn’t just about stitching  amino acids together -it’s about ensuring those sequences fold into stable, functional structures. A crucial part of evaluating ProtGPT2’s performance involves assessing whether the sequences it generates resemble natural proteins not only in composition but also in their three-dimensional structure and stability.

This section explores how well ProtGPT2 performs in structure prediction, folding stability, flexibility, and function retention — all essential for biologically viable proteins.

### 7.1 Folding Stability: Insights from AlphaFold and Rosetta
To understand how well ProtGPT2’s sequences fold, researchers used AlphaFold, a state-of-the-art structure prediction tool. The key metric here is pLDDT—a confidence score where higher values indicate more reliable and ordered structures. 

Remarkably:
-**Natural proteins** showed high structural confidence, with a mean pLDDT of 75.3, and 66% of sequences scoring above the 70 threshold.
- **ProtGPT2 sequences** followed closely behind, achieving a mean of 63.2 for their best predicted structures and 37% above the pLDDT 70 threshold.
- **Random sequences**, by contrast, had poor folding performance, averaging a pLDDT of only 44, with just 7.4% scoring above 70.

This means ProtGPT2 is not simply generating arbitrary strings—it’s producing sequences that AlphaFold believes can fold into well-structured proteins.

![](images/Rosetta vs MD.jpg)
<div style="text-align: center;">Figure 4: a. Average Rosetta energy units per residue for the three datasets. b. Root mean square deviation (RMSD) distribution for each MD dataset as computed by averaging RMSDs independently for each trajectory, represented as a boxplot.  </div>

To further confirm stability, the Rosetta Relax protocol was applied, simulating how these proteins would minimize energy to find their most stable forms. As seen in Figure 3a, both ProtGPT2 and natural sequences showed favorable energy scores (−1.90 and −1.73 REU/residue, respectively), while random sequences were significantly less stable (0.13 REU/residue). This underscores that ProtGPT2’s proteins are thermodynamically plausible.

### 7.2  Structural Flexibility: Are ProtGPT2 Proteins Dynamic?
Proteins in nature are not rigid—they flex and shift to carry out their functions. To examine whether ProtGPT2 sequences mimic this natural behavior, researchers performed molecular dynamics simulations, measuring RMSD (root mean square deviation) to capture structural fluctuations.

As shown in Figure 3b, the RMSD (root mean square deviation), which captures how much a protein fluctuates during simulation, was nearly the same for natural (2.93 Å) and ProtGPT2 (3.12 Å) sequences—both much lower than the RMSD of random sequences (9.41 Å). 

These results suggest that ProtGPT2’s sequences not only fold well but also move like real proteins, a key trait for functionality.

## 8. Exploring the Uncharted Protein Space
While many models attempt to mimic known protein structures, one of ProtGPT2’s most exciting capabilities lies in its potential to go beyond — to generate proteins that reside in previously uncharted regions of the protein landscape. This section explores how ProtGPT2 pushes the boundaries of structural biology through novel folds and preserved function.

### 8.1 Network Embedding of Protein Space

To understand how novel the generated proteins really are, researchers visualized the structural space occupied by ProtGPT2-generated sequences. Using a network-based representation, they connected sequences from two datasets:
- SCOPe 2.07 (manually curated structural database)
- ProtGPT2 outputs (generated de novo)

Edges in the network were formed based on alignment similarity, allowing visualization of how closely or distantly ProtGPT2 sequences relate to known protein folds.
![](<images/Protein_Space.jpg>)
<div style="text-align: center;">Network of structural relationships between ProtGPT2-generated and natural proteins. Each dot represents a protein. White nodes are ProtGPT2 sequences connected to known structural classes. </div>

Key highlights:
- The combined network had 59,612 nodes and over 427,000 edges.
- Over 50% of nodes clustered in one large component — with ProtGPT2 sequences often acting as bridges between disconnected islands in the structural landscape.
- Six diverse proteins were selected to visualize ProtGPT2’s reach into multiple SCOP classes: all-α, all-β, α/β, α+β, membrane proteins and small proteins.

ProtGPT2 doesn’t just replicate known structures — it expands the protein universe by generating sequences that connect distant regions, indicating potential for novel structural and functional exploration.

## 8.2 Preserving Function: Binding Site Conservation

Another striking feature of ProtGPT2 is its ability to retain functional relevance. To evaluate this, researchers used FoldSeek to superimpose two ProtGPT2-generated sequences with their closest structural homologs from the PDB (Protein Data Bank). Their goal was to check if key ligand-binding residues were preserved — a critical feature for biological function.

![](<images/Binding_Site.jpg>)
<div style="text-align: center;">Predicted ProtGPT2 structures (left) aligned with natural proteins (right). Ligand-binding residues are shown in stick representation. Matching residues highlight functional conservation.</div>

**Observation**
- Sequence 357 matched a blue-light sensor (1X0P_A), preserving 3 binding residues exactly and 2 with chemically similar substitutions.
- Sequence 475 mimicked a phosphodiesterase (5M1T_A), retaining 3 of 5 binding residues with minimal deviations.

Despite a low global sequence identity (~30%), ProtGPT2 zero-shot generated sequences that preserved critical interaction hotspots, suggesting strong potential for functional protein design — even without task-specific supervision.

## 9. Critical Perspective and Insights
The ProtGPT2 study represents a bold stride in harnessing the generative capabilities of large language models for de novo protein design. By adapting a GPT2-like architecture to the protein sequence domain and rigorously validating its outputs across structural, functional, and dynamic benchmarks, the authors make a compelling case for the use of LLMs beyond natural language. However, while the findings are promising, it is crucial to examine the broader implications, technical limitations, and future directions for improvement.

### 9.1 Strengths of the Study
One of the standout strengths of the paper is its multi-layered evaluation pipeline. Rather than stopping at syntactic plausibility or sequence diversity, the authors go further — validating generated proteins using:

- AlphaFold2 for structure prediction
- Rosetta for energy minimization
- Molecular dynamics for functional dynamics
- Homology searches using HHblits and FoldSeek

These orthogonal methods show that ProtGPT2 sequences are not just plausible, but structurally stable and biologically meaningful.

Another important contribution is the integration of generated sequences into the protein similarity network. ProtGPT2 bridges previously unconnected regions and populates “dark” areas of protein space, indicating novel topological potential — including folds with no known structural equivalent (e.g., protein 4266).

The model also demonstrates rapid scalability, generating tens of thousands of sequences with relatively low computational cost post-training — a useful trait for large-scale library design.

### 9.2 Limitations

Despite the promising results, some important limitations emerge:

1. **No Explicit Functional Optimization**: While some sequences preserved binding residues, ProtGPT2 is unguided — there is no direct control over desired activity or substrate specificity.

2. **Biophysical Awareness is Limited**: BPE-based tokenization, while efficient, doesn’t align with the residue-level semantics of proteins. It may obscure features critical for folding, binding, or structural compatibility.

3. **No Experimental Validation**: All assessments are in silico — AlphaFold predictions, Rosetta energy, MD simulations. Without wet-lab testing, real-world utility remains speculative.

4. **Interpretability**: ProtGPT2 functions as a black box — we lack insight into why specific motifs or residues are selected, making it harder to debug or tailor outputs.

### Areas for Improvement
To enhance ProtGPT2’s practical design capabilities, several areas offer room for improvement:

- Conditional Generation: Prompt-guided output (e.g., “design a calcium-binding alpha helix”) would allow goal-directed generation.

- Chemical Constraint Integration: Including folding rules, disulfide formation, transmembrane constraints, or post-translational modifications would reduce unrealistic or unstable designs.

- Real-Time Interpretability Tools: Visualization of attention, motif tracking, or embedding space distances could help users understand and refine outputs.

- Pretraining with Functional Datasets: Fine-tuning on curated protein families (e.g., enzymes, receptors) could help the model learn function-relevant motifs.

### 9.4 Future Directions
This study opens several exciting avenues for advancing AI-driven protein design:

- Multimodal Feedback Loops: Integrate ProtGPT2 with AlphaFold or molecular docking tools during generation for “structure- and function-aware” sampling.

- High-Throughput Experimental Validation: Use phage display, yeast surface display, or synthetic biology tools to rapidly test generated libraries.

- Task-Specific Transfer Learning: Fine-tune on specific folds or tasks (e.g., kinase inhibitors, antimicrobial peptides) to bias outputs toward real-world use cases.

- Foundation Models in Biology: ProtGPT2 shows the viability of GPT-style models as general-purpose sequence designers, suggesting the future of GPT for Proteins, akin to GPT-4 in language.

- Controlled Diversity: Incorporate sampling strategies that balance novelty and reliability (e.g., top-k + temperature + repetition penalties) to optimize for innovation without loss of structure.

## Final Thoughts
ProtGPT2 is not a replacement for rational design or evolutionary selection. Yet, its ability to generate realistic, diverse, and foldable proteins in seconds opens new possibilities in protein engineering, drug discovery, and synthetic biology. With further development—particularly in conditioning, functional targeting, and experimental feedback—ProtGPT2 may evolve into an indispensable assistant for molecular designers. It brings us closer to a future where protein sequences can be written like code—customized, creative, and computationally guided.
