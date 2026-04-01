# Subjective Retrieval Quality Analysis

This report presents sample retrieval results for qualitative analysis, consolidating the findings from the three-tier evaluation framework (Tier 1: per-domain BEIR, Tier 2: routing accuracy, Tier 3: cross-domain fusion).

Queries are selected to cover a range of scenarios: single-domain specialist queries, interdisciplinary cross-domain queries, and expected-failure cases.

---

## xd_01: "CRISPR gene editing treatment for hereditary disease"

**Expected domains:** biomedical, medical  
**Routing decision:** medical (conf: 0.60), scidocs (conf: 0.19)
  
**Active domains queried:** general, medical  
**Domains in top-5:** general, medical  
**Latency:** 2871.6 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | general | 0.0000 | (no title) | deletion on a chromosome. in genetics, a deletion ( also called gene deletion, deficiency, or deletion mutation ) ( sign : i´ ) is a mutation ( a gene… |
| 2 | general | 0.0000 | (no title) | genome editing. This factsheet is also available as a pdf (with images not included below). The term genome editing comprises a range of molecular tec… |
| 3 | medical | 0.0000 | China approves first gene therapy | China approves first gene therapy |
| 4 | general | 0.0000 | (no title) | Genetic engineering, also called genetic modification, is the direct manipulation of an organism's genome using biotechnology. It is a set of technolo… |
| 5 | medical | 0.0000 | Therapy of Ebola and Marburg virus infections | Therapy of Ebola and Marburg virus infections |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## xd_05: "algorithmic trading strategy design using reinforcement learning"

**Expected domains:** finance, science  
**Routing decision:** scidocs (conf: 0.99)
  
**Active domains queried:** general, scidocs  
**Domains in top-5:** general, scidocs  
**Latency:** 455.2 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | general | 0.0000 | (no title) | Â© Wiley 2010 * Review of Learning Objectives Define the role of Business Strategy Explain how a Business strategy is developed Explain the role of Op… |
| 2 | scidocs | 0.0000 | Algorithms for Inverse Reinforcement Learning | Algorithms for Inverse Reinforcement Learning |
| 3 | general | 0.0000 | (no title) | A heuristic technique (/ h j ÊÉËr Éª s t Éª k /; Ancient Greek: Îµá½ÏÎ¯ÏÎºÏ, find or discover), often called simply a heuristic, is any approac… |
| 4 | scidocs | 0.0000 | Fuzzy Passive-Aggressive classification: A robust and effici… | Fuzzy Passive-Aggressive classification: A robust and efficient algorithm for online classification problems |
| 5 | general | 0.0000 | (no title) | A heuristic technique, often called simply a heuristic, is any approach to problem solving, learning, or discovery that employs a practical method not… |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## xd_10: "transformer models for scientific literature review automation"

**Expected domains:** science, biomedical, scidocs  
**Routing decision:** scidocs (conf: 0.99)
  
**Active domains queried:** general, scidocs  
**Domains in top-5:** general, scidocs  
**Latency:** 163.7 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | general | 0.0000 | (no title) | A transformer is an electrical device that transfers electrical energy between two or more circuits through electromagnetic induction. A varying curre… |
| 2 | scidocs | 0.0000 | SYSTEM DYNAMICS : SYSTEMIC FEEDBACK MODELING FOR POLICY ANAL… | SYSTEM DYNAMICS : SYSTEMIC FEEDBACK MODELING FOR POLICY ANALYSIS |
| 3 | general | 0.0000 | (no title) | A transformer changes the voltage of an electric current while transferring it from one circuit to another through electromagnetic induction. Fluctuat… |
| 4 | scidocs | 0.0000 | Fast Effective Rule Induction | Fast Effective Rule Induction |
| 5 | general | 0.0000 | (no title) | The most studied transform fault in the world is the San Andreas Fault hope this helped <3 |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## xd_12: "reinforcement learning for personalised drug dosing"

**Expected domains:** medical, science, biomedical  
**Routing decision:** scidocs (conf: 0.99)
  
**Active domains queried:** general, scidocs  
**Domains in top-5:** general, scidocs  
**Latency:** 144.3 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | general | 0.0000 | (no title) | In popular use, positive reinforcement is often used as a synonym for reward, with people (not behavior) thus being reinforced, but this is contrary t… |
| 2 | scidocs | 0.0000 | Algorithms for Inverse Reinforcement Learning | Algorithms for Inverse Reinforcement Learning |
| 3 | general | 0.0000 | (no title) | Positive Reinforcement: Positive reinforcement is a very powerful and effective tool to help shape and change behavior. Positive reinforcement works b… |
| 4 | scidocs | 0.0000 | Bayesian Learning for Neural Networks | Bayesian Learning for Neural Networks |
| 5 | general | 0.0000 | (no title) | Second they examine the ability of social learning theory to explain these behaviors. The study revolved around the concepts of differential associati… |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## xd_22: "autonomous vehicle sensor fusion and decision algorithms"

**Expected domains:** science, general  
**Routing decision:** scidocs (conf: 0.99)
  
**Active domains queried:** general, scidocs  
**Domains in top-5:** general, scidocs  
**Latency:** 158.4 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | scidocs | 0.0000 | Team MIT Urban Challenge Technical Report | darpa site visit course. experimental results demonstrate all basic navigation and some basic traffic behaviors, including unoccupied autonomous drivi… |
| 2 | general | 0.0000 | (no title) | Applications of GMR GMR sensors find a wide range of applications: Fast and accurate position and motion sensing of mechanical components in precision… |
| 3 | scidocs | 0.0000 | Photogrammetric Multi-View Stereo and Imaging Network Design | Photogrammetric Multi-View Stereo and Imaging Network Design |
| 4 | general | 0.0000 | (no title) | Applications GMR sensors find a wide range of applications: Fast and accurate position and motion sensing of mechanical components in precision engine… |
| 5 | scidocs | 0.0000 | Unmanned surface vehicles: An overview of developments and c… | Unmanned surface vehicles: An overview of developments and challenges |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## Overall Observations

> TODO: Write a brief paragraph (4–6 sentences) summarising system strengths and limitations observed across all five queries. Reference specific examples from the queries above.
