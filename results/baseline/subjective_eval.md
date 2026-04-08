# Subjective Retrieval Quality Analysis

This report presents sample retrieval results for qualitative analysis, consolidating the findings from the three-tier evaluation framework (Tier 1: per-domain BEIR, Tier 2: routing accuracy, Tier 3: cross-domain fusion).

Queries are selected to cover a range of scenarios: single-domain specialist queries, interdisciplinary cross-domain queries, and expected-failure cases.

---

## xd_01: "How does machine learning improve drug discovery in pharmaceutical research?"

**Expected domains:** biomedical, science, general  
**Routing decision:** biomedical (conf: 0.50), medical (conf: 0.23)
  
**Active domains queried:** general, biomedical  
**Domains in top-5:** biomedical  
**Latency:** 556.0 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | biomedical | 0.0000 | Natural products as sources of new drugs over the last 25 ye… | compound approved as a drug in this 25 plus year time frame. we wish to draw the attention of readers to the rapidly evolving recognition that a signi… |
| 2 | biomedical | 0.0000 | Plant phenolics as drug leads -- what is missing? | evidenced by examples of endogenous phenols, phytochemicals containing aryl hydroxyl groups and phenolic synthetic drugs. it is postulated that applic… |
| 3 | biomedical | 0.0000 | At the crossroad of lifespan, calorie restriction, chromatin… | the sirtuins as major players in cellular homeostasis and human diseases that act through a whole range of biochemical substrates and physiological pr… |
| 4 | biomedical | 0.0000 | From beans to berries and beyond: teamwork between plant che… | from beans to berries and beyond : teamwork between plant chemicals for protection of optimal human health. it is now well known to consumers around t… |
| 5 | biomedical | 0.0000 | The drugs don’t work? antidepressants and the current and fu… | the drugs don ’ t work? antidepressants and the current and future pharmacological management of depression depression is a potentially life - threate… |

**Analysis:**
> Routing correctly identified biomedical as the primary domain, though the expected domain "science" was not queried. Results are tangentially relevant — they discuss drug compounds and pharmacology, but none specifically address machine learning applications in drug discovery. This suggests the biomedical corpus (NFCorpus) lacks ML-focused biomedical content. Actionable: the "general" domain (MS MARCO) would likely contain better results for this cross-cutting query, and it was correctly included via always_include_general.

---

## xd_05: "How does blockchain technology affect financial regulation?"

**Expected domains:** finance, general  
**Routing decision:** finance (conf: 0.98)
  
**Active domains queried:** general, finance  
**Domains in top-5:** finance  
**Latency:** 367.8 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | finance | 0.0000 | (no title) | payout plans if, based on the first - round results, they think it will reject them in the second round. related article regulators back trump on loos… |
| 2 | finance | 0.0000 | (no title) | many of which were linked to subprime mortgages and spread and amplified losses in the u. s. housing market. one breed of cdos are on a comeback path … |
| 3 | finance | 0.0000 | (no title) | goodman recently pointed out that investors are shunning us bonds and notes ; the lack of other buyers forced the federal reserve to buy “ a stunning.… |
| 4 | finance | 0.0000 | (no title) | paying for them • wave of downgrades in 2nd half of 2007 a result of unexpectedly poor performance of subprime mortgages originated in 2006 o attribut… |
| 5 | finance | 0.0000 | (no title) | we don ’ t see that demand from our clients and we wouldn ’ t recommend it, ” said markus stadlmann, chief investment officer at lloyds private bankin… |

**Analysis:**
> Routing correctly identified finance (0.98 conf). However, results are about subprime mortgages and CDOs — related to finance broadly but not to blockchain or financial regulation. The finance FAISS index was built with the wrong encoder (ProsusAI/finbert, a sentiment classifier), so dense retrieval is essentially random. Once the index is rebuilt with BAAI/bge-base-en-v1.5, relevance should improve significantly. Actionable: rebuild finance index with correct retrieval encoder.

---

## xd_10: "What methods detect antibiotic resistance in hospital settings?"

**Expected domains:** medical, biomedical  
**Routing decision:** biomedical (conf: 0.50), medical (conf: 0.34)
  
**Active domains queried:** general, biomedical  
**Domains in top-5:** biomedical  
**Latency:** 308.7 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | biomedical | 0.0000 | Salmonella enterica serotype Enteritidis: increasing inciden… | are commonly isolated from humans and chickens. conclusions : most se infections in the united states are acquired from domestic sources, and the prob… |
| 2 | biomedical | 0.0000 | Antimicrobial-resistant and extraintestinal pathogenic Esche… | by multivariate analysis, beef or pork and poultry from natural - food stores exhibited reduced risks of e. coli contamination and antimicrobial resis… |
| 3 | biomedical | 0.0000 | Ancestral antibiotic resistance in Mycobacterium tuberculosi… | profiling analyses demonstrate that whib7 transcription determines drug resistance by activating expression of a regulon including genes involved in r… |
| 4 | biomedical | 0.0000 | Rapid Detection of Extended-Spectrum-β-Lactamase-Producing E… | - m producers. a few esbl producers ( n = 16 ) that remained susceptible to cefotaxime were not detected. the test was also evaluated on spiked blood … |
| 5 | biomedical | 0.0000 | Clostridium difficile infection in the community: a zoonotic… | clostridium difficile infection in the community : a zoonotic disease? clostridium difficile infections ( cdis ) are traditionally seen in elderly and… |

**Analysis:**
> Routing identified biomedical (0.50) and medical (0.34) — both relevant domains for antibiotic resistance. Results from biomedical (NFCorpus) are topically relevant, covering antimicrobial resistance in Salmonella, E. coli, M. tuberculosis, ESBL-producing bacteria, and C. difficile. This is a strong retrieval success despite the non-retrieval encoder (BioBERT). The medical domain was not included as a separate active domain due to confidence distribution, though "general" was included via always_include_general. Actionable: consider lowering confidence_threshold to include medical domain for queries where top-2 domains are both specialist.

---

## xd_12: "What causes inflation and how does it affect consumer spending?"

**Expected domains:** finance, general  
**Routing decision:** finance (conf: 0.98)
  
**Active domains queried:** general, finance  
**Domains in top-5:** finance  
**Latency:** 187.5 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | finance | 0.0000 | (no title) | yield curve's positive slope, and a steepening yield curve could indicate an increasing government deficit, declines in private savings, or reduced ca… |
| 2 | finance | 0.0000 | (no title) | the money before the price changed on them. unexpected inflation may cause a temporary dip in spending until wages adjust, however consumers still nee… |
| 3 | finance | 0.0000 | (no title) | interest rates and inflation is the main reason for fluctuating rates in the first place. the fed will tend to raise rates to try to slow inflation, a… |
| 4 | finance | 0.0000 | (no title) | the us look relatively attractive and so there is huge demand for us dollars and bonds. any significant move in us interest rates risks driving to dol… |
| 5 | finance | 0.0000 | (no title) | individuals and businesses. this potential decline in investment could lead to decreased revenue / profits for businesses, which could in turn cause d… |

**Analysis:**
> Routing correctly identified finance (0.98 conf). Results are highly relevant — discussing inflation, interest rates, consumer spending, and yield curves. Result #2 explicitly mentions "unexpected inflation may cause a temporary dip in spending until wages adjust," directly answering the query. This is one of the system's strongest results despite the stale finance index. Actionable: finance queries show reasonable topical matching even with the wrong encoder, suggesting FiQA corpus content is well-aligned with financial queries.

---

## xd_22: "How does natural language processing improve financial sentiment analysis?"

**Expected domains:** finance, science  
**Routing decision:** finance (conf: 0.98)
  
**Active domains queried:** general, finance  
**Domains in top-5:** finance  
**Latency:** 177.7 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | finance | 0.0000 | (no title) | payout plans if, based on the first - round results, they think it will reject them in the second round. related article regulators back trump on loos… |
| 2 | finance | 0.0000 | (no title) | according to financial consultancy coalition. clo investors have been handsomely rewarded in recent months. j. p. morgan strategist rishad ahluwalia r… |
| 3 | finance | 0.0000 | (no title) | we don ’ t see that demand from our clients and we wouldn ’ t recommend it, ” said markus stadlmann, chief investment officer at lloyds private bankin… |
| 4 | finance | 0.0000 | (no title) | s. banks, believes that & amp ; quot ; higher interest rates would be a net plus for the community banking sector that would help them extend more cre… |
| 5 | finance | 0.0000 | (no title) | ##bunking the naysayers who claim that unfunded pension liabilities will obliterate municipal and state budgets. " " & gt ; even a year of good return… |

**Analysis:**
> Routing correctly identified finance (0.98 conf), but results are entirely about traditional finance topics (bank regulations, CLOs, interest rates) — none mention NLP or sentiment analysis. This is a cross-domain query that needs both finance and CS/science content, but the system only retrieves from finance+general. The finance corpus (FiQA) contains user questions about investing, not academic NLP research. Actionable: this highlights a fundamental limitation of domain-specific corpora — cross-cutting queries spanning academic CS and applied finance have no appropriate corpus to draw from.

---

## Overall Observations

> The routing system shows high confidence and generally correct domain assignment — finance queries (xd_05, xd_12, xd_22) are all correctly routed to finance with 0.98 confidence, and biomedical queries (xd_01, xd_10) are correctly identified. However, routing accuracy alone does not guarantee retrieval quality: the finance FAISS index is built with a non-retrieval encoder (ProsusAI/finbert), leading to topically random results for xd_05 despite correct routing. The strongest result was xd_12 (inflation query), where multiple results directly addressed the query intent. The weakest was xd_22 (NLP + finance), which highlights the system's inability to serve truly cross-domain queries that span academic CS and applied finance. The biomedical domain (xd_01, xd_10) demonstrates that even with a non-retrieval encoder (BioBERT), topical relevance can be achieved for domain-specific queries, though precision is low. A clear priority for optimization is rebuilding finance and general domain indexes with retrieval-trained encoders.
