# Subjective Retrieval Quality Analysis

This report presents sample retrieval results for qualitative analysis, consolidating the findings from the three-tier evaluation framework (Tier 1: per-domain BEIR, Tier 2: routing accuracy, Tier 3: cross-domain fusion).

Queries are selected to cover a range of scenarios: single-domain specialist queries, interdisciplinary cross-domain queries, and expected-failure cases.

---

## xd_01: "How does machine learning improve drug discovery in pharmaceutical research?"

**Expected domains:** biomedical, science, general  
**Routing decision:** scidocs (conf: 0.73), medical (conf: 0.19)
  
**Active domains queried:** general, scidocs  
**Domains in top-5:** scidocs  
**Latency:** 49289.8 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | scidocs | 6.0478 | Low Data Drug Discovery with One-Shot Learning | low data drug discovery with one - shot learning recent advances in machine learning have made significant contributions to drug discovery. deep neura… |
| 2 | scidocs | 2.4122 | Seq2seq Fingerprint: An Unsupervised Deep Molecular Embeddin… | seq2seq fingerprint : an unsupervised deep molecular embedding for drug discovery many of today's drug discoveries require expertise knowledge and ins… |
| 3 | scidocs | 2.3271 | Chemi-net: a graph convolutional network for accurate drug p… | Chemi-net: a graph convolutional network for accurate drug property prediction Absorption, distribution, metabolism, and excretion (ADME) studies are … |
| 4 | scidocs | 1.8668 | Graph Convolutional Neural Networks for ADME Prediction in D… | Graph Convolutional Neural Networks for ADME Prediction in Drug Discovery ADME in-silico methods have grown increasingly powerful over the past twenty… |
| 5 | scidocs | 0.7156 | Machine learning in cardiovascular medicine: are we there ye… | ##s for ai in medicine are the development of automated risk prediction algorithms which can be used to guide clinical care ; use of unsupervised lear… |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## xd_05: "How does blockchain technology affect financial regulation?"

**Expected domains:** finance, general  
**Routing decision:** finance (conf: 1.00)
  
**Active domains queried:** general, finance  
**Domains in top-5:** finance, general  
**Latency:** 15752.6 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | finance | 0.0139 | (no title) | the fact that you don't know that the chain doesn't only contain hashes concerns me...  I've consulted with several companies who are bringing blockch… |
| 2 | finance | -0.0300 | (no title) | the blockchain introduces scarcity to the internet. BTC is nothing but the first mover and a storage of value.   ETH has a huge real-world application… |
| 3 | finance | -0.3479 | (no title) | ##chain hype stood the test of time? or does the next phase in the gartner hype cycle — “ the trough of disillusionment ” — beckon? we are either ther… |
| 4 | general | -0.5973 | (no title) | Because transactional records using blockchain technology are software based, they can also record simple programs and algorithms on which smart contr… |
| 5 | finance | -0.7565 | (no title) | Every major financial firm and government in the world is looking at blockchain technology right now. It's definitely revolutionary. I think it's the … |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## xd_10: "What methods detect antibiotic resistance in hospital settings?"

**Expected domains:** medical, biomedical  
**Routing decision:** medical (conf: 0.76), biomedical (conf: 0.20)
  
**Active domains queried:** general, medical  
**Domains in top-5:** medical  
**Latency:** 26146.3 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | medical | 4.2608 | Epidemiology of Multi-Drug Resistant Organisms in a Teaching… | epidemiology of multi - drug resistant organisms in a teaching hospital in oman : a one - year hospital - based study background. antimicrobial resist… |
| 2 | medical | 3.3429 | Performance and impact of a multiplex PCR in ICU patients wi… | performance and impact of a multiplex pcr in icu patients with ventilator - associated pneumonia or ventilated hospital - acquired pneumonia backgroun… |
| 3 | medical | 2.5275 | Role of plasmid carrying bla(NDM) in mediating antibiotic re… | ##olates. intriguingly, bla ( ndm ) - positive isolates also showed a high degree of resistance to antibiotics of different classes. the potential co … |
| 4 | medical | 1.8455 | Outpatient Antibiotic Stewardship: A Growing Frontier—Combin… | outpatient antibiotic stewardship : a growing frontier — combining myxovirus resistance protein a with other biomarkers to improve antibiotic use back… |
| 5 | medical | 1.1130 | Assessing the intestinal carriage rates of vancomycin-resist… | ##pectrometry and antibiotic susceptibilities were tested by e - test. vancomycin resistance genes were detected by polymerase chain reaction. medical… |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## xd_12: "What causes inflation and how does it affect consumer spending?"

**Expected domains:** finance, general  
**Routing decision:** finance (conf: 1.00)
  
**Active domains queried:** general, finance  
**Domains in top-5:** finance, general  
**Latency:** 8822.9 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | finance | 7.3463 | (no title) | the money before the price changed on them. unexpected inflation may cause a temporary dip in spending until wages adjust, however consumers still nee… |
| 2 | general | 6.1639 | (no title) | What causes negative inflation or deflation? Deflation, or negative inflation, happens when prices fall because the supply of goods is higher than the… |
| 3 | general | 5.2932 | (no title) | By Investopedia | November 14, 2014 â 1:11 PM EST. A: Deflation, or negative inflation, happens when prices fall because the supply of goods is high… |
| 4 | finance | 4.0120 | (no title) | inflation can go up for a number of reasons. boom times can cause inflation, as everyone is making and spending a lot of money, so prices and inflatio… |
| 5 | finance | 2.9982 | (no title) | i do appreciate the explanation. i was being a bit facetious about hotdogs in that i don't buy the same hotdog over and over again. but seriously, i s… |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## xd_22: "How does natural language processing improve financial sentiment analysis?"

**Expected domains:** finance, science  
**Routing decision:** finance (conf: 0.99)
  
**Active domains queried:** general, finance  
**Domains in top-5:** finance, general  
**Latency:** 7181.7 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | general | -0.5534 | (no title) | Natural language processing (NLP) is at the root of this complicated mission. The ability to analyze and extract meaning from narrative text or other … |
| 2 | finance | -2.9252 | (no title) | My wild ass guess is that it will be a pair of voice controlled, connected headphones. *smartphones*, if you will, that aren't dependent on a phone in… |
| 3 | finance | -4.1195 | (no title) | I think you definitely have a shot then.  There are people who come from a wide variety of backgrounds at my firm, not strictly finance people and not… |
| 4 | finance | -5.0895 | (no title) | on quantopian called'sentiment analyzer'( or something like that ) which basically quantifies sentiment around any given security using web scrapers t… |
| 5 | finance | -6.1391 | (no title) | There's been some work on sentiment analysis and it's effect on stock prices. Would be interesting to see a model that uses sentiment as well as more … |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## Overall Observations

> TODO: Write a brief paragraph (4–6 sentences) summarising system strengths and limitations observed across all five queries. Reference specific examples from the queries above.
