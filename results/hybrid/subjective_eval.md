# Subjective Retrieval Quality Analysis

This report presents sample retrieval results for qualitative analysis, consolidating the findings from the three-tier evaluation framework (Tier 1: per-domain BEIR, Tier 2: routing accuracy, Tier 3: cross-domain fusion).

Queries are selected to cover a range of scenarios: single-domain specialist queries, interdisciplinary cross-domain queries, and expected-failure cases.

---

## xd_01: "How does machine learning improve drug discovery in pharmaceutical research?"

**Expected domains:** biomedical, science, general  
**Routing decision:** biomedical (conf: 0.50), medical (conf: 0.23)
  
**Active domains queried:** general, biomedical  
**Domains in top-5:** biomedical, general  
**Latency:** 12481.5 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | general | 0.0000 | (no title) | Pharmaceutical company. A pharmaceutical company, or drug company, is a commercial business licensed to research, develop, market and/or distribute dr… |
| 2 | biomedical | 0.0000 | Death by design: where curcumin sensitizes drug-resistant tu… | ##lux pumps, ( ii ) enhanced drug detoxification, ( iii ) rapid dna repair efficiency, ( iv ) defects in apoptosis regulation, and ( v ) active cell s… |
| 3 | general | 0.0000 | (no title) | Transfer of Learning is ensuring what people learn on training programs is transferred into real business results. Learn how transfer of learning enab… |
| 4 | biomedical | 0.0000 | At the crossroad of lifespan, calorie restriction, chromatin… | the sirtuins as major players in cellular homeostasis and human diseases that act through a whole range of biochemical substrates and physiological pr… |
| 5 | general | 0.0000 | (no title) | The path to drug addiction begins with the voluntary act of taking drugs. But over time, a person's ability to choose not to do so becomes compromised… |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## xd_05: "How does blockchain technology affect financial regulation?"

**Expected domains:** finance, general  
**Routing decision:** finance (conf: 0.98)
  
**Active domains queried:** general, finance  
**Domains in top-5:** finance, general  
**Latency:** 8902.3 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | general | 0.0000 | (no title) | Because transactional records using blockchain technology are software based, they can also record simple programs and algorithms on which smart contr… |
| 2 | finance | 0.0000 | (no title) | ##chain hype stood the test of time? or does the next phase in the gartner hype cycle — “ the trough of disillusionment ” — beckon? we are either ther… |
| 3 | general | 0.0000 | (no title) | Battle of the blockchains: From Bitcoin to Lisk, five different types of blockchain explained Dominic Powell / Thursday, December 7, 2017 A common mis… |
| 4 | finance | 0.0000 | (no title) | " this is the best tl ; dr i could make, [ original ] ( https : / / hackernoon. com / cryptoeconomics - paving - the - future - of - blockchain - tech… |
| 5 | general | 0.0000 | (no title) | Different types of Blockchain available in Market these are Public Blockchain, Private Blockchain, and Consortium Blockchain. There are various Blockc… |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## xd_10: "What methods detect antibiotic resistance in hospital settings?"

**Expected domains:** medical, biomedical  
**Routing decision:** biomedical (conf: 0.50), medical (conf: 0.34)
  
**Active domains queried:** general, biomedical  
**Domains in top-5:** biomedical, general  
**Latency:** 1860.6 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | general | 0.0000 | (no title) | The results show that the antimicrobial resistance patterns of the causes of UTI are highly variable and continuous surveillance of trends in resistan… |
| 2 | biomedical | 0.0000 | Ancestral antibiotic resistance in Mycobacterium tuberculosi… | profiling analyses demonstrate that whib7 transcription determines drug resistance by activating expression of a regulon including genes involved in r… |
| 3 | general | 0.0000 | (no title) | Antibiotic sensitivity testing; Antimicrobial susceptibility testing. After the culture (sample) is collected from a person, it is sent to a lab. Ther… |
| 4 | biomedical | 0.0000 | Food Animals and Antimicrobials: Impacts on Human Health | unfavorable consequences of ntas is not clear to all stakeholders. substantial data show elevated antibiotic resistance in bacteria associated with an… |
| 5 | general | 0.0000 | (no title) | Urinary tract infection. requiring antibiotic treatment is. defined in the general population. as the presence of significant. amounts of a single mic… |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## xd_12: "What causes inflation and how does it affect consumer spending?"

**Expected domains:** finance, general  
**Routing decision:** finance (conf: 0.98)
  
**Active domains queried:** general, finance  
**Domains in top-5:** finance, general  
**Latency:** 3692.4 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | general | 0.0000 | (no title) | What causes negative inflation or deflation? Deflation, or negative inflation, happens when prices fall because the supply of goods is higher than the… |
| 2 | finance | 0.0000 | (no title) | the money before the price changed on them. unexpected inflation may cause a temporary dip in spending until wages adjust, however consumers still nee… |
| 3 | general | 0.0000 | (no title) | What can cause price deflation? Learn what price deflation is, how inflation rates can be calculated using the consumer price index and what some caus… |
| 4 | finance | 0.0000 | (no title) | of the state of the economy, and how surprised the people talking about the numbers are. sometimes they refer to inflation since the last month, and t… |
| 5 | general | 0.0000 | (no title) | Inflation rates vary from year to year and from currency to currency. Since 1950, the U.S. dollar inflation rate, as measured by the December-to-Decem… |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## xd_22: "How does natural language processing improve financial sentiment analysis?"

**Expected domains:** finance, science  
**Routing decision:** finance (conf: 0.98)
  
**Active domains queried:** general, finance  
**Domains in top-5:** finance, general  
**Latency:** 1359.1 ms

| Rank | Domain | Score | Title | Snippet |
|------|--------|-------|-------|---------|
| 1 | general | 0.0000 | (no title) | Natural language processing (NLP) is at the root of this complicated mission. The ability to analyze and extract meaning from narrative text or other … |
| 2 | finance | 0.0000 | (no title) | There's been some work on sentiment analysis and it's effect on stock prices. Would be interesting to see a model that uses sentiment as well as more … |
| 3 | general | 0.0000 | (no title) | Ruby is used in typical scripting language applications such as text processing and glue or middleware programs. It's suitable for small, ad-hoc scrip… |
| 4 | finance | 0.0000 | (no title) | I'm a programmer, and worked for a company doing natural language analysis and parsing.  Yes, it is possible to figure out what reviews are more valua… |
| 5 | general | 0.0000 | (no title) | Purpose & Importance of Financial Statements can be analyzed in the context of users of financial statements and their respective interests. The objec… |

**Analysis:**
> TODO: Write 2–3 sentences analysing: (1) correctness of the routing decision, (2) relevance of top results, (3) any notable failures or surprises, (4) one actionable observation.

---

## Overall Observations

> TODO: Write a brief paragraph (4–6 sentences) summarising system strengths and limitations observed across all five queries. Reference specific examples from the queries above.
