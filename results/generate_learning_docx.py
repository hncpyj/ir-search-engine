"""
generate_learning_docx.py

Builds a comprehensive IR / RAG / NLP learning document targeted at an
INTJ learning style: first principles, derivations, why-before-how,
systematic structure, depth over breadth, no fluff.
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

NAVY = RGBColor(0x1F, 0x3A, 0x68)
DARK = RGBColor(0x2C, 0x3E, 0x50)
RED  = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x1E, 0x82, 0x49)
GREY = RGBColor(0x6C, 0x75, 0x7D)


def H1(doc, text):
    h = doc.add_heading(text, level=1)
    for r in h.runs:
        r.font.color.rgb = NAVY


def H2(doc, text):
    h = doc.add_heading(text, level=2)
    for r in h.runs:
        r.font.color.rgb = DARK


def H3(doc, text):
    h = doc.add_heading(text, level=3)


def P(doc, text):
    return doc.add_paragraph(text)


def WHY(doc, text):
    """A 'Why this matters' callout — INTJs need motivation before mechanics."""
    p = doc.add_paragraph()
    r = p.add_run("Why this matters: ")
    r.bold = True
    r.font.color.rgb = GREEN
    p.add_run(text)


def KEY(doc, text):
    """Key insight callout."""
    p = doc.add_paragraph()
    r = p.add_run("Key insight: ")
    r.bold = True
    r.font.color.rgb = RED
    p.add_run(text)


def FORMULA(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(10)
    p.paragraph_format.left_indent = Cm(1)


def CODE(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    p.paragraph_format.left_indent = Cm(0.5)


def BULLET(doc, text):
    doc.add_paragraph(text, style="List Bullet")


def NUM(doc, text):
    doc.add_paragraph(text, style="List Number")


# ────────────────────────────────────────────────────────────────────────────
# Document construction
# ────────────────────────────────────────────────────────────────────────────

def title_page(doc):
    doc.add_paragraph()
    doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Information Retrieval, RAG & NLP")
    r.bold = True
    r.font.size = Pt(28)
    r.font.color.rgb = NAVY

    s = doc.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = s.add_run("A First-Principles Learning Guide for IR Engineers")
    r.font.size = Pt(16)
    r.font.color.rgb = GREY

    doc.add_paragraph()
    m = doc.add_paragraph()
    m.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = m.add_run(
        "Designed for systematic, depth-first learners.\n"
        "Every concept derived from why before how.\n\n"
        "From TF-IDF to dense retrieval, hybrid systems, transformers, and RAG."
    )
    r.font.size = Pt(11)
    r.font.color.rgb = GREY

    doc.add_page_break()


def how_to_read(doc):
    H1(doc, "How to Read This Guide")
    P(doc,
      "This guide is structured for depth-first learners who prefer to understand "
      "the why before the how. Each concept is introduced by motivating the problem "
      "it solves, then derived from first principles, then connected to neighbouring "
      "ideas, and finally placed in a system context.")

    H2(doc, "Reading order")
    P(doc,
      "Read sequentially the first time. Each chapter builds on prior chapters: "
      "you cannot understand BM25 without TF-IDF, you cannot understand dense retrieval "
      "without word embeddings, and you cannot understand RAG without retrieval.")

    H2(doc, "Three-pass reading strategy")
    NUM(doc, "Pass 1: Read for the structure. Skip the formulas. Build a mental map of how the parts connect.")
    NUM(doc, "Pass 2: Read for the derivations. Work out each formula with pen and paper.")
    NUM(doc, "Pass 3: Read for the system context. Notice the trade-offs and where each technique fits.")

    H2(doc, "Conventions")
    BULLET(doc, "WHY callouts motivate each concept before introducing it.")
    BULLET(doc, "KEY callouts highlight the one insight you must remember.")
    BULLET(doc, "Formulas are in monospace; derive them yourself, do not memorise.")

    doc.add_page_break()


def chapter_1(doc):
    H1(doc, "Chapter 1 — What Is Information Retrieval, Really?")

    H2(doc, "1.1 The core problem")
    P(doc,
      "Given a query and a corpus of documents, return the documents most likely "
      "to satisfy the user's information need, ranked by likelihood of relevance.")
    P(doc, "Three things in that sentence are doing all the work:")
    BULLET(doc, "Information need — what the user actually wants, which is rarely what they typed.")
    BULLET(doc, "Likelihood of relevance — a probabilistic concept, not a binary one.")
    BULLET(doc, "Ranked — order matters more than membership.")

    WHY(doc,
        "Every IR technique you will learn is an attempt to better estimate "
        "the probability that a document is relevant to an information need, given a query. "
        "If you keep this framing in mind, every formula in this guide will feel inevitable rather than arbitrary.")

    H2(doc, "1.2 IR vs database lookup")
    P(doc,
      "A database query is exact: SELECT * WHERE id = 42 either matches or does not. "
      "IR is approximate: 'climate change effects' must rank a document about 'global warming impact' "
      "highly even though no query term appears in it. The fundamental shift is from exact matching to similarity.")

    H2(doc, "1.3 The four eras of IR")
    NUM(doc, "Boolean era (1960s–70s): exact term matching with AND/OR/NOT. Brittle but interpretable.")
    NUM(doc, "Statistical era (1970s–2000s): TF-IDF, BM25, language models. Probabilistic ranking.")
    NUM(doc, "Learning-to-rank era (2000s–2010s): use ML to combine hundreds of features.")
    NUM(doc, "Neural era (2018–today): bi-encoders, cross-encoders, dense vectors, RAG.")

    KEY(doc,
        "Each era did not replace the previous one. Modern systems combine boolean filters, "
        "BM25 lexical scoring, dense retrieval, and learned rerankers. You need all four.")

    H2(doc, "1.4 The two questions you will answer in every system")
    NUM(doc, "How do I represent text so that similarity can be computed? (Encoding question)")
    NUM(doc, "Given billions of documents, how do I find the top-k similar ones fast? (Indexing question)")
    P(doc,
      "Every chapter in this guide is about either the encoding question or the indexing question, "
      "or how the two interact.")

    doc.add_page_break()


def chapter_2(doc):
    H1(doc, "Chapter 2 — Text Preprocessing: Making Text Comparable")

    WHY(doc,
        "Two documents that are semantically identical can look completely different at the byte level. "
        "Preprocessing closes this gap so that downstream similarity measures actually measure similarity.")

    H2(doc, "2.1 Tokenisation")
    P(doc,
      "Splitting a string into tokens. Sounds trivial; it is not. The choices you make here propagate "
      "through every later stage.")
    BULLET(doc, "Whitespace tokenisation: 'data-driven' becomes one token. Loses information.")
    BULLET(doc, "Punctuation-aware: 'data-driven' becomes ['data', 'driven']. Better for English.")
    BULLET(doc, "Subword (BPE, WordPiece): 'tokenisation' becomes ['token', '##isation']. Handles unknown words.")
    BULLET(doc, "Character-level: every character is a token. Robust to typos but loses semantics.")

    H2(doc, "2.2 Normalisation")
    BULLET(doc, "Case folding: 'Apple' and 'apple' become the same token. Loses proper-noun signal.")
    BULLET(doc, "Unicode NFKC: 'café' (composed) and 'café' (decomposed) become the same byte sequence.")
    BULLET(doc, "Diacritic removal: 'résumé' becomes 'resume'. Useful for cross-lingual matching, dangerous otherwise.")

    H2(doc, "2.3 Stemming vs lemmatisation")
    P(doc, "Both reduce inflected words to a canonical form, but they differ in how:")
    BULLET(doc, "Stemming: rule-based truncation. 'running' → 'run', 'happiness' → 'happi'. Fast, lossy.")
    BULLET(doc, "Lemmatisation: dictionary-based. 'running' → 'run', 'better' → 'good'. Slow, accurate.")

    KEY(doc,
        "Stemming and lemmatisation only matter for lexical (BM25) systems. Dense retrieval encoders "
        "learn morphological variation directly from data and need neither.")

    H2(doc, "2.4 Stopwords")
    P(doc,
      "Words like 'the', 'of', 'and' appear in nearly every document. They contain almost no "
      "information about relevance. Classical systems remove them to save space and reduce noise. "
      "Neural systems keep them — context-aware models extract syntactic information from them.")

    H2(doc, "2.5 The decision tree for preprocessing")
    NUM(doc, "Are you using a neural encoder? → Use the encoder's tokenizer. Stop here.")
    NUM(doc, "Are you using BM25? → Lowercase, tokenise on punctuation, remove stopwords, stem.")
    NUM(doc, "Multilingual? → Keep diacritics, use language-specific tokenisers, skip stemming.")

    doc.add_page_break()


def chapter_3(doc):
    H1(doc, "Chapter 3 — Term Frequency: The First Honest Attempt")

    H2(doc, "3.1 The naive idea")
    P(doc,
      "If a query term appears more often in a document, the document is probably more about that term. "
      "Count occurrences. Done.")

    FORMULA(doc, "tf(t, d) = number of times term t appears in document d")

    H2(doc, "3.2 Why raw TF fails")
    P(doc, "Three problems become obvious within minutes of trying it:")
    NUM(doc,
        "Document length bias. A 10,000-word document will have higher term counts than a 100-word document "
        "even when both are equally relevant. The longer document wins for the wrong reason.")
    NUM(doc,
        "Saturation. A document that mentions 'climate' 50 times is not 50 times more relevant than one that "
        "mentions it once. After a few mentions, additional occurrences add little information.")
    NUM(doc,
        "Common-word dominance. Every document mentions 'the' hundreds of times. Raw TF makes 'the' look "
        "like the most important term in every document.")

    H2(doc, "3.3 First fixes")
    P(doc, "The classical IR field iterated through these in chronological order:")
    FORMULA(doc, "Length normalised:  tf(t, d) / |d|")
    FORMULA(doc, "Log dampening:      1 + log(tf(t, d))   if tf > 0, else 0")
    FORMULA(doc, "Augmented:          0.5 + 0.5 * tf(t, d) / max_tf(d)")

    P(doc,
      "Each fix addresses one of the three problems but introduces new ones. Length normalisation "
      "over-penalises long documents that genuinely contain more information. Log dampening helps with "
      "saturation. Augmented normalisation handles both length and saturation but throws away absolute counts.")

    KEY(doc,
        "TF alone is a one-dimensional view: how often. We also need: how rare. Without rarity, "
        "the most frequent words drown out the most informative ones. This is what IDF will fix in Chapter 4.")

    H2(doc, "3.4 What you should remember")
    BULLET(doc, "TF measures within-document importance.")
    BULLET(doc, "Raw TF is too noisy; always dampen and normalise.")
    BULLET(doc, "TF cannot distinguish 'relevant' from 'frequent'. You need a second signal.")

    doc.add_page_break()


def chapter_4(doc):
    H1(doc, "Chapter 4 — Inverse Document Frequency: Why We Need It and How to Derive It")

    H2(doc, "4.1 The motivating problem")
    P(doc,
      "Consider the query 'quantum entanglement'. Both terms appear in a relevant physics paper. "
      "But 'quantum' might appear in 50,000 documents in your corpus while 'entanglement' might appear in 800. "
      "Intuitively, 'entanglement' is doing more work to identify the relevant document — it is rarer, "
      "and matching it is more informative.")

    WHY(doc,
        "We need a way to weight rare matches more than common matches. IDF is exactly that weight, "
        "and it falls out naturally from information theory.")

    H2(doc, "4.2 Derivation from information theory")
    P(doc,
      "Information theory tells us that the information content of an event with probability p is "
      "-log(p). Rare events carry more information; certain events carry zero.")
    P(doc, "Apply this to terms. Define:")

    FORMULA(doc, "p(t) = probability that a randomly chosen document contains term t")
    FORMULA(doc, "     = df(t) / N")

    P(doc, "where df(t) is the document frequency (number of documents containing t) and N is the total documents.")
    P(doc, "The information content of seeing term t in a document is:")

    FORMULA(doc, "I(t) = -log(p(t)) = -log(df(t) / N) = log(N / df(t))")

    P(doc,
      "That is exactly the IDF formula. It is not an arbitrary weight — it is the self-information "
      "of the event 'this document contains term t'. Rare terms have high IDF because seeing them is surprising; "
      "common terms have low IDF because seeing them is unsurprising.")

    H2(doc, "4.3 The formula")
    FORMULA(doc, "idf(t) = log(N / df(t))")
    P(doc, "Practical concerns force two adjustments:")
    BULLET(doc,
           "Smoothing: if df(t) = 0, log diverges. Add 1: idf(t) = log(N / (1 + df(t))) or log((N + 1) / df(t)).")
    BULLET(doc,
           "Boundedness: if df(t) = N (term appears in every document), idf = 0, which is correct — the term "
           "carries no information about which document is relevant.")

    H2(doc, "4.4 Converting TF into TF-IDF")
    P(doc,
      "TF measures within-document importance. IDF measures across-corpus discriminativeness. "
      "Multiplying them gives a weight that is high when a term is both frequent in this document "
      "and rare across the corpus:")

    FORMULA(doc, "tf-idf(t, d) = tf(t, d) * idf(t)")

    P(doc, "More precisely, with the practical dampening and smoothing applied:")

    FORMULA(doc, "tf-idf(t, d) = (1 + log(tf(t, d))) * log((N + 1) / (df(t) + 1))")

    H2(doc, "4.5 A worked example")
    P(doc,
      "Corpus N = 10,000 documents. Query: 'quantum entanglement'.")
    BULLET(doc, "df('quantum') = 5,000 → idf = log(10000/5000) = log(2) ≈ 0.69")
    BULLET(doc, "df('entanglement') = 100 → idf = log(10000/100) = log(100) ≈ 4.61")
    P(doc,
      "If both terms appear once in document A, the TF-IDF contribution of 'entanglement' is "
      "about 6.7× greater than 'quantum'. The rarer term dominates the score, which is exactly what we want.")

    KEY(doc,
        "IDF is not a hack. It is the self-information of the event 'this term appears in this document'. "
        "Once you see this derivation, the formula stops being arbitrary and starts being inevitable.")

    H2(doc, "4.6 IDF variants you will see in the wild")
    BULLET(doc, "Plain: log(N / df)")
    BULLET(doc, "Smoothed: log((N + 1) / (df + 1)) + 1  (scikit-learn default)")
    BULLET(doc, "Probabilistic: log((N - df + 0.5) / (df + 0.5))  (used in BM25)")
    P(doc,
      "The probabilistic IDF can go negative for very common terms (df > N/2), which BM25 clamps to 0. "
      "The intuition: terms that appear in more than half the corpus are anti-informative.")

    doc.add_page_break()


def chapter_5(doc):
    H1(doc, "Chapter 5 — The Vector Space Model and Cosine Similarity")

    H2(doc, "5.1 Documents as vectors")
    P(doc,
      "Once each (term, document) pair has a TF-IDF weight, a document becomes a vector in a high-dimensional "
      "space — one dimension per vocabulary term. A 100k-vocabulary corpus produces 100k-dimensional vectors. "
      "Most entries are zero (a document only contains a few hundred unique words), so we store them sparsely.")

    H2(doc, "5.2 Why cosine, not Euclidean")
    P(doc,
      "Suppose document A is the same as document B but repeated 10 times. In Euclidean space, A and B are "
      "10× apart. In cosine space, they are identical — the angle between their vectors is zero. "
      "We almost always want the cosine view: similarity should be invariant to document length.")

    FORMULA(doc, "cos(q, d) = (q · d) / (||q|| * ||d||)")

    P(doc,
      "After L2 normalisation (dividing each vector by its magnitude), cosine similarity reduces to a dot product. "
      "This is why dense retrieval libraries always normalise vectors before indexing — it lets the FAISS dot-product "
      "search compute cosine similarity for free.")

    H2(doc, "5.3 The geometric intuition")
    P(doc,
      "Imagine the vocabulary axis 'climate' and the vocabulary axis 'finance'. A climate science paper points "
      "along the climate axis. A finance paper points along the finance axis. A paper about climate finance "
      "points at 45 degrees between them. Cosine similarity measures the angle, which is exactly what 'topical "
      "overlap' should mean.")

    KEY(doc,
        "Once you learn to think of documents as points in a high-dimensional space, every retrieval technique "
        "becomes a question about geometry: nearest neighbours, hyperplanes, projections, clusters.")

    doc.add_page_break()


def chapter_6(doc):
    H1(doc, "Chapter 6 — BM25: Why TF-IDF Wasn't Enough")

    WHY(doc,
        "TF-IDF was the state of the art for two decades, but it has two structural flaws that BM25 fixes. "
        "BM25 is still, in 2026, the strongest single-model lexical baseline and the BM25 component of every "
        "hybrid system you will build.")

    H2(doc, "6.1 The two flaws TF-IDF didn't address")
    NUM(doc,
        "TF saturation is linear-in-log: doubling the term count keeps adding score. In reality, the 20th "
        "occurrence of 'climate' is much less informative than the 2nd. We need a hard saturation curve.")
    NUM(doc,
        "Length normalisation is too aggressive (dividing by length) or too weak (no normalisation at all). "
        "We need a tuneable middle ground that penalises long documents only somewhat.")

    H2(doc, "6.2 The BM25 formula, derived")
    P(doc, "BM25 fixes both with two parameters: k1 (saturation) and b (length normalisation).")

    FORMULA(doc,
            "BM25(q, d) = Σ_t  idf(t) * (tf(t, d) * (k1 + 1)) / "
            "                (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))")

    P(doc, "Pull this apart piece by piece:")
    BULLET(doc,
           "idf(t) — same as before, but using the probabilistic IDF: log((N - df + 0.5) / (df + 0.5)).")
    BULLET(doc,
           "tf * (k1 + 1) / (tf + k1 * ...) — this is a saturating function. As tf grows, it asymptotes "
           "to k1 + 1. Typical k1 = 1.2 to 2.0. Lower k1 means faster saturation.")
    BULLET(doc,
           "(1 - b + b * |d| / avgdl) — the length normalisation factor. b = 0 means no normalisation, "
           "b = 1 means full normalisation, b = 0.75 (default) is the empirically optimal middle ground.")

    H2(doc, "6.3 Why BM25 works so well")
    P(doc,
      "The saturation curve matches human relevance perception: the first few occurrences of a term carry most "
      "of the signal, additional ones add little. The length normalisation accounts for the fact that longer "
      "documents have more chances to contain a term by accident, but also genuinely contain more information. "
      "b = 0.75 is the sweet spot between these two effects.")

    H2(doc, "6.4 BM25 in 2026")
    P(doc,
      "Despite seven years of dense retrieval research, BM25 remains competitive on out-of-domain evaluation "
      "(BEIR benchmark) because it has no training data assumptions. Dense retrievers can be catastrophically "
      "bad on domains they were not trained on; BM25 is consistently mediocre, which often makes it the better "
      "choice when you cannot retrain. This is the main reason every production hybrid system still includes BM25.")

    KEY(doc,
        "Memorise the BM25 formula. You will use it for the rest of your career, and the parameters k1 and b "
        "are the first two knobs you will tune in any hybrid system.")

    doc.add_page_break()


def chapter_7(doc):
    H1(doc, "Chapter 7 — Inverted Indexes: Making Search Scalable")

    WHY(doc,
        "BM25 is mathematically elegant but useless if you have to scan every document for every query. "
        "The inverted index is the data structure that makes lexical IR practical at web scale. "
        "Understanding it is non-negotiable: even if you only ever use FAISS, you will be asked about it in interviews.")

    H2(doc, "7.1 The forward index (what NOT to do)")
    P(doc, "A forward index maps each document to its terms:")
    CODE(doc,
         "doc_1 → [climate, change, effects, agriculture]\n"
         "doc_2 → [climate, finance, investment]\n"
         "doc_3 → [global, warming, oceans]")
    P(doc,
      "Querying for 'climate' requires scanning every document. O(N) per query. Useless beyond a few thousand documents.")

    H2(doc, "7.2 The inverted index")
    P(doc, "Invert the mapping: each term points to the documents containing it.")
    CODE(doc,
         "climate → [doc_1, doc_2]\n"
         "change  → [doc_1]\n"
         "finance → [doc_2]\n"
         "global  → [doc_3]\n"
         "warming → [doc_3]")
    P(doc,
      "Querying for 'climate' is O(1) lookup followed by reading the postings list. The postings list is sorted "
      "by document ID so we can intersect lists for AND queries with a merge join.")

    H2(doc, "7.3 What lives in a postings list")
    P(doc, "Production indexes store more than document IDs:")
    BULLET(doc, "Document ID (sorted, delta-encoded for compression).")
    BULLET(doc, "Term frequency in that document (for BM25 scoring).")
    BULLET(doc, "Position offsets (for phrase queries: 'climate change' as a phrase).")
    BULLET(doc, "Field information (title vs body).")

    H2(doc, "7.4 Scoring algorithms: WAND and MaxScore")
    P(doc,
      "Even with an inverted index, computing BM25 for every document containing any query term is too slow "
      "for short queries on large corpora. WAND (Weak AND) and MaxScore are early-termination algorithms that "
      "skip documents that cannot make it into the top-k. They can be 10× to 100× faster than naive scoring.")
    P(doc,
      "You will not implement these from scratch — Lucene, Tantivy, and PISA do it for you — but you should "
      "know they exist so you understand why search engines are fast.")

    H2(doc, "7.5 The trade-off vs dense retrieval")
    BULLET(doc, "Inverted index: exact term matching, no embedding cost, very fast for selective queries.")
    BULLET(doc, "Dense vector index: semantic matching, embedding cost per query, slower but more recall on paraphrases.")
    P(doc, "Modern systems use both. Chapter 13 explains how they fuse.")

    doc.add_page_break()


def chapter_8(doc):
    H1(doc, "Chapter 8 — Word Embeddings: From Symbols to Geometry")

    WHY(doc,
        "TF-IDF and BM25 cannot match 'car' with 'automobile'. They treat words as opaque symbols with no "
        "internal structure. Word embeddings are the breakthrough that gave words a geometry: similar words "
        "occupy nearby points in a continuous vector space. Everything that came after — BERT, RAG, LLMs — "
        "rests on this idea.")

    H2(doc, "8.1 The distributional hypothesis")
    P(doc,
      "Firth, 1957: 'You shall know a word by the company it keeps.' Two words that appear in similar contexts "
      "have similar meanings. 'Car' and 'automobile' appear with the same neighbours: 'drive', 'engine', "
      "'wheel', 'highway'. This is enough to learn that they mean the same thing — without ever consulting a dictionary.")

    H2(doc, "8.2 Word2Vec — the first practical embedding")
    P(doc,
      "Mikolov et al., 2013. Train a shallow neural network to predict either:")
    BULLET(doc, "Skip-gram: given a word, predict its context words.")
    BULLET(doc, "CBOW: given context words, predict the centre word.")
    P(doc,
      "After training, throw away the network. The hidden layer weights for each word ARE the word embedding: "
      "a 300-dimensional vector. Geometrically meaningful: the famous 'king - man + woman ≈ queen' analogy emerges naturally.")

    H2(doc, "8.3 GloVe — the matrix factorisation alternative")
    P(doc,
      "Pennington et al., 2014. Build the global word-word co-occurrence matrix and factorise it into low-rank "
      "embeddings. Mathematically equivalent to Word2Vec under certain conditions but trains faster on small corpora.")

    H2(doc, "8.4 What you can do with word embeddings")
    BULLET(doc, "Synonym lookup: nearest neighbours of 'happy' are 'joyful', 'glad', 'pleased'.")
    BULLET(doc, "Document representation by averaging: bag-of-embeddings is a strong baseline.")
    BULLET(doc, "Cross-lingual alignment: train English and French embeddings separately, then learn a rotation.")

    H2(doc, "8.5 What word embeddings cannot do")
    P(doc,
      "Each word has exactly one vector. 'Bank' (river) and 'bank' (financial institution) collapse into the same point. "
      "Polysemy is fundamentally invisible to word2vec/GloVe. This is the limitation that contextual embeddings (BERT) fix.")

    KEY(doc,
        "Word embeddings replaced symbols with geometry. Once you can compute distances between words, "
        "every NLP problem becomes a geometry problem — and decades of geometric algorithms become available.")

    doc.add_page_break()


def chapter_9(doc):
    H1(doc, "Chapter 9 — Transformers and Attention from First Principles")

    WHY(doc,
        "BERT, GPT, BGE, every embedding model you will use — they are all transformers. You cannot have "
        "an opinion about which encoder to choose, why fine-tuning works, or why attention scales quadratically "
        "in sequence length unless you understand the attention mechanism.")

    H2(doc, "9.1 The problem attention solves")
    P(doc,
      "Pre-2017 sequence models (RNNs, LSTMs) processed words one at a time in order. Long-range dependencies "
      "(matching a verb to its subject 30 words away) had to flow through 30 hidden states without being forgotten. "
      "It rarely worked.")
    P(doc,
      "Attention fixes this by letting every word directly look at every other word in the sequence. "
      "Information no longer has to travel through intermediate hidden states.")

    H2(doc, "9.2 The mechanism, derived")
    P(doc, "For each word in the input, compute three vectors by linear projection:")
    BULLET(doc, "Query (Q): what this word is looking for.")
    BULLET(doc, "Key (K): what this word offers.")
    BULLET(doc, "Value (V): the content this word will pass on if selected.")
    P(doc, "The attention score between word i and word j is the dot product of Q_i and K_j. "
           "Higher dot product = more attention.")

    FORMULA(doc, "Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V")

    P(doc, "Read it left to right:")
    NUM(doc, "Q K^T computes a similarity matrix between every query and every key.")
    NUM(doc, "Divide by sqrt(d_k) to keep gradients stable when d_k is large.")
    NUM(doc, "Softmax turns scores into a probability distribution over keys.")
    NUM(doc, "Multiply by V to take a weighted sum of value vectors.")

    P(doc,
      "The output for each position is a weighted average of all positions, where the weights are learned "
      "from the data. This is content-based addressing: each position decides what to pay attention to "
      "based on what it needs.")

    H2(doc, "9.3 Multi-head attention")
    P(doc,
      "Run several attention layers in parallel with different Q/K/V projections, concatenate the outputs. "
      "Different heads learn to attend to different relationships: one head might track syntactic dependencies, "
      "another semantic similarity, another coreference. Empirically, 8 to 16 heads is the sweet spot.")

    H2(doc, "9.4 The transformer block")
    P(doc, "A single transformer layer is:")
    NUM(doc, "Multi-head self-attention.")
    NUM(doc, "Add and normalise (residual connection plus layer norm).")
    NUM(doc, "Feed-forward network (two linear layers with a ReLU or GeLU between them).")
    NUM(doc, "Add and normalise.")
    P(doc, "Stack 12 of these (BERT-base) or 24 (BERT-large) and you have an encoder. Stack 96 and you have GPT-3.")

    H2(doc, "9.5 Why quadratic")
    P(doc,
      "The attention matrix is N × N where N is the sequence length. Doubling the input length quadruples "
      "the memory and compute. This is why most retrieval encoders cap input at 512 tokens — beyond that, "
      "memory blows up. Long-context models (Longformer, BigBird, FlashAttention) reduce this in various ways "
      "but the trade-off remains.")

    KEY(doc,
        "Attention is just learned soft retrieval: each position retrieves a weighted combination of all other "
        "positions based on query-key similarity. Transformers are stacks of differentiable retrievers.")

    doc.add_page_break()


def chapter_10(doc):
    H1(doc, "Chapter 10 — BERT and Contextual Embeddings")

    H2(doc, "10.1 The contextual breakthrough")
    P(doc,
      "Word2Vec gives the word 'bank' a single vector. BERT gives it a different vector depending on the surrounding "
      "sentence: 'I sat on the river bank' produces a different vector than 'I deposited money at the bank'. "
      "The vector is a function of the context, not just the word.")

    H2(doc, "10.2 How BERT is trained")
    P(doc, "Two self-supervised objectives, both designed to need no labels:")
    NUM(doc,
        "Masked Language Modelling (MLM): randomly mask 15% of input tokens, train the model to predict them "
        "from context. Forces the model to encode bidirectional context into every token's representation.")
    NUM(doc,
        "Next Sentence Prediction (NSP): given two sentences, predict whether they were consecutive in the corpus. "
        "Later research showed this objective is mostly useless; modern variants (RoBERTa, BGE) drop it.")

    H2(doc, "10.3 What you get from a trained BERT")
    P(doc,
      "For each input token, a 768-dimensional vector that captures both the token's identity and its context. "
      "For sequence classification tasks, BERT prepends a special [CLS] token whose final-layer vector is "
      "trained to summarise the whole sequence.")

    H2(doc, "10.4 Why you cannot use raw BERT for retrieval")
    P(doc,
      "BERT was trained on MLM and NSP, not on similarity. The [CLS] vector of 'cat sat on the mat' is "
      "not particularly close to the [CLS] vector of 'feline rested on the rug'. To get a useful retrieval encoder, "
      "you must fine-tune BERT with a contrastive objective. This is what Sentence-BERT does.")

    KEY(doc,
        "Domain-specific MLMs like BioBERT, SciBERT, ClinicalBERT are trained on the right vocabulary but with "
        "the wrong objective. They produce near-random retrieval scores. Use a retrieval-trained encoder "
        "(BGE, E5, GTE) on a general corpus and you will outperform a domain-specific MLM almost every time. "
        "This is the single most important practical lesson from the project that produced this guide.")

    doc.add_page_break()


def chapter_11(doc):
    H1(doc, "Chapter 11 — Bi-Encoders, Cross-Encoders, and Sentence-BERT")

    WHY(doc,
        "These are the two architectures that dominate neural retrieval. Choosing between them is the "
        "highest-stakes architecture decision in any RAG system: the wrong choice will either be too slow "
        "(cross-encoder on millions of documents) or too inaccurate (bi-encoder for the final ranking).")

    H2(doc, "11.1 Bi-encoder: encode independently, compare with dot product")
    P(doc,
      "The query is encoded by one BERT pass into a single vector. Each document is encoded by another BERT pass "
      "into another vector. Similarity is the dot product. Crucially, document vectors can be precomputed and "
      "stored in a vector index — the query is the only thing encoded at search time.")

    FORMULA(doc, "score(q, d) = encode(q) · encode(d)")

    H2(doc, "11.2 Cross-encoder: encode query and document together")
    P(doc,
      "Concatenate query and document into a single sequence with a [SEP] token, run through BERT, "
      "feed the [CLS] vector into a binary classifier. The model can attend across query and document tokens "
      "directly, which gives much higher accuracy than bi-encoders.")

    FORMULA(doc, "score(q, d) = classifier(BERT([CLS] q [SEP] d))")

    H2(doc, "11.3 The trade-off")
    P(doc, "Suppose you have 1 million documents and want to find the top 10 for a query.")
    BULLET(doc,
           "Bi-encoder: encode the query (1 BERT pass), look up nearest neighbours in a precomputed FAISS index "
           "(microseconds). Total: about 30 milliseconds.")
    BULLET(doc,
           "Cross-encoder: encode (query, document) pairs for all 1 million documents. 1 million BERT passes. "
           "Total: about 8 hours per query. Unusable for retrieval.")

    KEY(doc,
        "Bi-encoders are fast but less accurate. Cross-encoders are accurate but slow. The standard solution "
        "is a two-stage cascade: bi-encoder retrieves top-100 candidates, then a cross-encoder reranks those 100. "
        "This gives you cross-encoder accuracy at bi-encoder latency.")

    H2(doc, "11.4 Sentence-BERT: making BERT a bi-encoder")
    P(doc,
      "Reimers and Gurevych, 2019. Take a pretrained BERT, add a mean-pooling layer on top to produce a "
      "sentence embedding, fine-tune with a contrastive objective on similarity-labelled data (SNLI, STS). "
      "The result: a bi-encoder that produces vectors actually useful for similarity. This single paper made "
      "production semantic search practical.")

    H2(doc, "11.5 ColBERT: late interaction as a middle ground")
    P(doc,
      "ColBERT (Khattab and Zaharia, 2020) keeps a separate vector for every token instead of pooling to one "
      "vector per document. Scoring computes max-similarity between each query token and each document token, "
      "then sums. Far more accurate than bi-encoders, far cheaper than cross-encoders, but at the cost of much "
      "larger indexes (every token, not every document).")

    doc.add_page_break()


def chapter_12(doc):
    H1(doc, "Chapter 12 — Approximate Nearest Neighbour Search and FAISS")

    WHY(doc,
        "A bi-encoder gives you a 768-dimensional vector per document. Finding the nearest neighbours of "
        "a query vector among 100 million such vectors with brute force takes seconds. ANN indexes get this "
        "to single-digit milliseconds, with a small accuracy cost. You will use FAISS or a similar library "
        "in every dense retrieval system.")

    H2(doc, "12.1 The brute force baseline")
    P(doc,
      "Exact search: compute the dot product between the query vector and every document vector, sort. "
      "O(N * d) where N is the corpus size and d is the dimension. For N = 10^8 and d = 768, this is "
      "about 10 billion floating point operations per query. Too slow.")

    H2(doc, "12.2 The two strategies for sublinear search")
    P(doc, "Every ANN index is built on one of two ideas:")
    NUM(doc,
        "Partitioning: divide the vector space into cells, only search the cells closest to the query. "
        "Examples: IVF (Inverted File), tree-based methods.")
    NUM(doc,
        "Graph traversal: build a graph where each vector is a node, edges connect vectors that are close. "
        "Search by greedy traversal from a random starting point. Example: HNSW.")

    H2(doc, "12.3 HNSW — the modern default")
    P(doc,
      "Hierarchical Navigable Small Worlds (Malkov and Yashunin, 2018). Build a multi-layer graph where "
      "the top layer has few nodes with long edges (for fast coarse navigation) and the bottom layer has all "
      "nodes with short edges (for fine-grained search). Query: start at the top, greedily walk to the closest "
      "neighbour, descend a layer, repeat. Empirically gives 95-99% recall at 10x to 100x the speed of brute force.")
    BULLET(doc, "M parameter: number of edges per node. Higher = more accurate but more memory. Typical: 16-32.")
    BULLET(doc, "efConstruction: search depth at index build time. Higher = better graph but slower build.")
    BULLET(doc, "efSearch: search depth at query time. Tunes the recall/latency trade-off.")

    H2(doc, "12.4 Product Quantisation (PQ) — for the largest indexes")
    P(doc,
      "When you have a billion vectors and 768 dimensions, even storing the vectors costs terabytes. "
      "PQ compresses each vector by splitting it into chunks and quantising each chunk to a code from a small "
      "codebook. A 768-d float32 vector (3 KB) becomes a 16-byte code. 200x compression, with a small accuracy hit.")
    P(doc, "Combined with IVF (IVF_PQ in FAISS), you can search a billion-vector index in milliseconds on a single machine.")

    H2(doc, "12.5 Choosing an index type")
    BULLET(doc, "Under 100k vectors: Flat (brute force). Faster than HNSW for small indexes, exact recall.")
    BULLET(doc, "100k - 10M vectors: HNSW. Best speed/recall trade-off in this range.")
    BULLET(doc, "10M - 1B vectors: IVF + PQ. Memory becomes the bottleneck.")
    BULLET(doc, "Over 1B: distributed FAISS or specialised systems (Vespa, Vald, ScaNN).")

    KEY(doc,
        "ANN indexes are recall-bounded approximations. You set a recall target (say 95%), tune efSearch "
        "until you hit it, and accept that the bottom 5% of true nearest neighbours are missed. For retrieval "
        "this is almost always fine because the missed results are barely-relevant.")

    doc.add_page_break()


def chapter_13(doc):
    H1(doc, "Chapter 13 — Hybrid Retrieval and Reciprocal Rank Fusion")

    H2(doc, "13.1 Why hybrid")
    P(doc,
      "BM25 finds documents that share exact terms with the query. Dense retrieval finds documents that share "
      "meaning with the query. These are different signals. Documents that score well on both are almost "
      "certainly relevant; documents that score on only one might still be relevant. Combining them recovers "
      "both kinds of matches and consistently outperforms either alone on most domains.")

    P(doc, "Empirical evidence from the project that produced this guide:")
    BULLET(doc, "On domains with retrieval-trained encoders: hybrid ≈ dense (BM25 contributes little).")
    BULLET(doc, "On domains with non-retrieval MLM encoders: hybrid >> dense (BM25 carries most of the weight).")
    BULLET(doc,
           "Cross-domain queries: hybrid increases domain coverage by +40% because BM25 surfaces "
           "exact-match candidates that the dense encoder misses.")

    H2(doc, "13.2 The score combination problem")
    P(doc,
      "BM25 scores are unbounded positive numbers. Dense scores are typically in [-1, 1] or [0, 1] depending "
      "on normalisation. You cannot just add them: BM25 will dominate. Two solutions:")
    NUM(doc, "Score normalisation: min-max scale both lists to [0, 1], then weighted sum. Brittle: depends on the score distribution of each query.")
    NUM(doc, "Rank fusion: ignore scores entirely, combine ranks. Robust: invariant to score scales.")

    H2(doc, "13.3 Reciprocal Rank Fusion (RRF)")
    P(doc, "Cormack et al., 2009. The simplest and surprisingly best rank fusion method.")

    FORMULA(doc, "RRF_score(d) = Σ_i  1 / (k + rank_i(d))")

    P(doc,
      "where rank_i(d) is the rank of document d in retriever i's results, and k is a smoothing constant "
      "(typically 60). Documents that appear high in multiple lists accumulate score; documents that appear "
      "in only one list get a smaller contribution.")

    H2(doc, "13.4 Why k = 60")
    P(doc,
      "The reciprocal in RRF gives diminishing returns: rank 1 contributes 1/61, rank 2 contributes 1/62, "
      "rank 100 contributes 1/161. The k constant controls how much the top of the list dominates. "
      "k = 60 was found empirically by Cormack and is the de facto default. Tuning rarely improves things "
      "by more than 0.01 nDCG.")

    H2(doc, "13.5 The two-stage fusion pattern")
    P(doc,
      "In a multi-domain system like the one this guide accompanies, fusion happens twice: first within each "
      "domain (combine BM25 and dense from the same corpus), then across domains (combine the results from "
      "different specialist indexes). RRF is closed under composition: you can fuse fused lists with no issues.")

    KEY(doc,
        "RRF is a one-line algorithm with no hyperparameters worth tuning. If you are not using it, you are "
        "either underperforming or paying a cost in tuning complexity. It should be your first hybrid baseline.")

    doc.add_page_break()


def chapter_14(doc):
    H1(doc, "Chapter 14 — Rerankers and Learning to Rank")

    H2(doc, "14.1 Why rerank")
    P(doc,
      "Retrieval finds 100 documents that are probably relevant. Reranking decides which 10 are actually most "
      "relevant. The retrieval stage optimises for recall (don't miss anything) at high speed. The reranking "
      "stage optimises for precision (top results must be excellent) at lower speed because it only sees 100 candidates.")

    H2(doc, "14.2 Cross-encoder rerankers")
    P(doc,
      "Run a BERT-style cross-encoder on each (query, candidate) pair. The model attends across both tokens, "
      "producing a relevance score that captures fine-grained interactions a bi-encoder cannot. Typical speedup: "
      "100 candidates × 50 ms per pair = 5 seconds total. On GPU this drops to about 200 ms.")
    P(doc,
      "Standard models: ms-marco-MiniLM-L-6-v2 (small, fast), ms-marco-electra (medium), bge-reranker-large (state of the art).")

    H2(doc, "14.3 ColBERT rerankers")
    P(doc,
      "Late-interaction reranking. More accurate than vanilla bi-encoders, faster than cross-encoders. Used "
      "as either a retriever or a reranker depending on index budget.")

    H2(doc, "14.4 LLM rerankers (the new frontier)")
    P(doc,
      "Use an instruction-tuned LLM to rerank candidates. Two patterns:")
    BULLET(doc, "Pointwise: 'On a scale of 0 to 10, how relevant is document X to query Y?' Slow but accurate.")
    BULLET(doc, "Listwise: 'Here are 20 documents. Output them in order of relevance.' Faster, can be more accurate because the LLM sees the whole list.")

    H2(doc, "14.5 Learning to rank")
    P(doc,
      "Pre-neural era technique still in production at Google, Amazon, Microsoft. Extract hundreds of features "
      "(BM25 score, click-through rate, freshness, document length, query length, term overlap, ...) and train "
      "a gradient-boosted tree (LambdaMART) to predict relevance. Less accurate than neural rerankers but "
      "extremely interpretable, fast, and trivial to combine with business rules.")

    KEY(doc,
        "The retrieve-then-rerank cascade is the dominant production pattern. Retrieval is broad and fast; "
        "reranking is narrow and accurate. Optimise them independently.")

    doc.add_page_break()


def chapter_15(doc):
    H1(doc, "Chapter 15 — Evaluation Metrics: What 'Good' Means")

    WHY(doc,
        "Every IR engineer has watched a retrieval change look great qualitatively but tank in offline evaluation, "
        "or vice versa. Choosing the right metric for your task is the single most consequential decision you "
        "make in IR — it determines what 'progress' even means.")

    H2(doc, "15.1 The qrels (relevance judgments)")
    P(doc,
      "A qrel is a (query, document, relevance grade) triple, typically labelled by humans. Binary qrels use "
      "0 / 1 (irrelevant / relevant). Graded qrels use 0-3 or 0-4 (irrelevant / marginally / relevant / highly relevant). "
      "Without qrels, you cannot compute any metric. Building a high-quality qrel set is more expensive than "
      "building a model.")

    H2(doc, "15.2 Precision and Recall")
    FORMULA(doc, "Precision@k = (relevant docs in top k) / k")
    FORMULA(doc, "Recall@k    = (relevant docs in top k) / (total relevant docs)")
    P(doc, "Precision asks 'are the top results relevant?'. Recall asks 'did we find the relevant documents?'. "
           "Both are needed but neither is enough alone.")

    H2(doc, "15.3 Why P/R isn't enough")
    P(doc,
      "Precision@10 = 0.5 means 5 of the top 10 are relevant. But it does not say whether the relevant ones "
      "are at positions 1-5 (great) or positions 6-10 (much worse). Order matters; precision ignores it.")

    H2(doc, "15.4 MRR (Mean Reciprocal Rank)")
    FORMULA(doc, "MRR = 1 / rank of first relevant document, averaged across queries")
    P(doc,
      "Optimised for question answering: how quickly does the user find a relevant result? If the first relevant "
      "result is at position 3, MRR = 0.33. Ignores everything after the first hit, which makes it the right "
      "metric for QA but wrong for browsing tasks.")

    H2(doc, "15.5 nDCG (Normalised Discounted Cumulative Gain)")
    P(doc, "The metric you should use unless you have a specific reason not to.")
    FORMULA(doc, "DCG@k = Σ_{i=1}^k  rel_i / log2(i + 1)")
    FORMULA(doc, "nDCG@k = DCG@k / IDCG@k")
    P(doc,
      "where rel_i is the relevance grade of the document at rank i, and IDCG@k is the DCG of the ideal ranking. "
      "The log discount means each lower position contributes less. Normalisation by IDCG bounds the score to "
      "[0, 1] regardless of how many relevant documents exist.")
    P(doc,
      "nDCG handles graded relevance (a 'highly relevant' doc at rank 1 contributes more than a 'marginally "
      "relevant' doc), discounts position appropriately, and is comparable across queries with different numbers "
      "of relevant documents. This is why it is the standard for IR research.")

    H2(doc, "15.6 MAP (Mean Average Precision)")
    P(doc,
      "Average the precision at every position where a relevant document appears, then average across queries. "
      "Older metric, mostly replaced by nDCG. Still common in TREC papers and ad hoc IR benchmarks.")

    H2(doc, "15.7 The BEIR benchmark")
    P(doc,
      "Thakur et al., 2021. A collection of 18 datasets across different domains (medical, scientific, financial, "
      "argumentation) used to evaluate the out-of-domain generalisation of retrieval models. Critical insight: "
      "models that win on MS MARCO often lose on BEIR. The BEIR benchmark is the closest thing we have to "
      "measuring how well a retriever will work on YOUR domain.")

    H2(doc, "15.8 Statistical significance")
    P(doc,
      "If your new model improves nDCG by 0.005 over baseline, is that real or noise? Use a paired bootstrap "
      "or permutation test on the per-query metric values. A change is meaningful if it is statistically "
      "significant AND larger than the typical query-to-query variance, which is usually around 0.05 for nDCG.")

    KEY(doc,
        "Pick one metric and optimise for it. Looking at five metrics simultaneously means you will always "
        "find one that confirms whatever you want to believe. nDCG@10 is the right default for most ranking tasks.")

    doc.add_page_break()


def chapter_16(doc):
    H1(doc, "Chapter 16 — Retrieval-Augmented Generation (RAG)")

    H2(doc, "16.1 The motivation")
    P(doc,
      "LLMs hallucinate. They were trained on a snapshot of the internet from 2023 (or whenever) and cannot "
      "tell you what happened yesterday. They do not know your private documents. Fine-tuning on your data is "
      "expensive and changes the model's behaviour in unpredictable ways. RAG solves all three problems: "
      "instead of teaching the model new facts, retrieve the facts at query time and let the model use them as context.")

    WHY(doc,
        "RAG is what makes LLMs production-usable for any domain that requires correctness, freshness, "
        "or proprietary knowledge. Every customer support bot, every internal Q&A system, every code assistant "
        "that searches your repo is some form of RAG.")

    H2(doc, "16.2 The RAG architecture")
    P(doc, "At its simplest, RAG is just retrieval feeding into generation:")
    NUM(doc, "User asks a question.")
    NUM(doc, "Retriever finds k relevant documents from a corpus.")
    NUM(doc, "Generator (LLM) produces an answer conditioned on the retrieved documents.")
    NUM(doc, "Optional: cite which documents the answer used.")
    P(doc, "Every RAG system is a variation on this template. The interesting questions are: how do you retrieve well, "
           "how do you chunk the corpus, how do you combine retrieval results with the prompt, and how do you evaluate the result.")

    H2(doc, "16.3 Parametric vs non-parametric memory")
    P(doc,
      "An LLM's weights are parametric memory: facts compressed into the network's parameters during training. "
      "A retrieval index is non-parametric memory: facts stored verbatim in documents and looked up on demand. "
      "Parametric memory is fast but frozen; non-parametric memory is slower but updatable. RAG combines them.")

    H2(doc, "16.4 Chunking strategies")
    P(doc,
      "You cannot put a 500-page PDF into a context window. You must split it into chunks before indexing.")
    BULLET(doc, "Fixed-size: every chunk is N tokens. Simple, can break sentences in half.")
    BULLET(doc, "Sentence-aware: split at sentence boundaries, accumulate until N tokens. Better.")
    BULLET(doc,
           "Semantic: use a model to find topic boundaries and split there. Best quality, slowest to build.")
    BULLET(doc, "Hierarchical: index small chunks for retrieval, return larger surrounding context to the LLM.")
    P(doc,
      "Empirical default: 200-500 tokens per chunk with 10-20% overlap between adjacent chunks. The overlap "
      "ensures information that lands on a chunk boundary is captured by both neighbours.")

    H2(doc, "16.5 Query reformulation and HyDE")
    P(doc,
      "Users often ask short, ambiguous questions. The retriever needs more signal. Two techniques:")
    BULLET(doc,
           "Query expansion: use an LLM to rewrite the query into multiple alternative phrasings, retrieve for each, fuse with RRF.")
    BULLET(doc,
           "HyDE (Hypothetical Document Embeddings): ask an LLM to write a hypothetical answer to the query, "
           "embed that, search for similar real documents. Counter-intuitive but effective: the hypothetical "
           "answer is closer in vector space to real answers than the original question is.")

    H2(doc, "16.6 RAG evaluation: Faithfulness vs answer relevance")
    P(doc, "Standard IR metrics measure whether the retrieved documents are relevant. RAG adds two new metrics:")
    BULLET(doc,
           "Faithfulness: does the generated answer actually use the retrieved documents, or did the LLM make it up? "
           "Measure by checking whether each claim in the answer is supported by the context.")
    BULLET(doc,
           "Answer relevance: does the answer address the question, regardless of whether it is correct? "
           "Measure by similarity between question and answer.")
    P(doc,
      "Both metrics can be computed with another LLM as a judge. Frameworks like RAGAS automate this. "
      "The tricky part is that LLM judges are themselves biased — be careful about interpreting absolute scores.")

    H2(doc, "16.7 Advanced patterns")
    BULLET(doc, "Self-RAG: the model decides when to retrieve. Avoids retrieval for queries it can answer directly.")
    BULLET(doc, "Agentic RAG: the model can issue multiple retrieval queries, refine, and iterate.")
    BULLET(doc, "GraphRAG: build a knowledge graph from the corpus, traverse relations during retrieval.")
    BULLET(doc, "Multi-hop: chain retrievals so each step depends on the previous step's results.")

    KEY(doc,
        "RAG is not a single technique. It is a design space. The retriever, the chunker, the prompt template, "
        "the reranker, the generator — each is independently tunable, and most of your engineering time will go "
        "into picking the right combination for your data.")

    doc.add_page_break()


def chapter_17(doc):
    H1(doc, "Chapter 17 — Query Routing and Multi-Domain Systems")

    H2(doc, "17.1 The problem")
    P(doc,
      "When your corpus spans multiple domains (medical, legal, financial, general web), a single global index "
      "performs worse than separate per-domain indexes for two reasons: different domains need different encoders, "
      "and a single index forces all queries to compete with all documents regardless of topical fit.")

    H2(doc, "17.2 The architecture")
    P(doc, "A multi-domain retrieval system has three components:")
    NUM(doc, "Per-domain indexes, each with its own encoder optimised for that domain.")
    NUM(doc, "A query classifier (router) that predicts which domain(s) a query belongs to.")
    NUM(doc, "A fusion stage that combines results from the active domains into a single ranking.")

    H2(doc, "17.3 Routing modes")
    BULLET(doc,
           "routed_only: send the query only to the top-1 predicted domain. Highest precision, lowest recall.")
    BULLET(doc,
           "routed_with_general: send the query to top-1 plus a 'general' fallback. Safer for low-confidence predictions.")
    BULLET(doc,
           "broadcast: send the query to all domains. Maximum recall, but introduces noise from irrelevant domains.")

    H2(doc, "17.4 The classifier")
    P(doc,
      "Typically a fine-tuned DistilBERT or similar small model. Trained on (query, domain) pairs. "
      "Top-1 accuracy is the headline metric, but top-2 accuracy is what matters in practice because you "
      "can fall back to the second prediction when the top-1 is uncertain.")

    H2(doc, "17.5 The classifier's failure modes")
    BULLET(doc,
           "Insufficient training data for some domains: the classifier collapses to 'predict the majority class'. "
           "Solved by oversampling minority domains or using class weights.")
    BULLET(doc,
           "Train/eval contamination: if your evaluation queries overlap with the training set, accuracy is "
           "inflated. Always use a held-out test set.")
    BULLET(doc,
           "Cross-domain queries: a query like 'economic impact of pandemic on healthcare' belongs to multiple "
           "domains. Hard classification gives the wrong answer. Solution: top-2 routing or broadcast for low-confidence cases.")

    KEY(doc,
        "Routing is a recall problem disguised as a classification problem. The cost of missing a domain "
        "(zero recall from that index) is much higher than the cost of including an extra one (some noise in "
        "the fused results). Always route generously and let RRF clean up the noise.")

    doc.add_page_break()


def chapter_18(doc):
    H1(doc, "Chapter 18 — Production Concerns")

    H2(doc, "18.1 The latency budget")
    P(doc, "A typical interactive search has about 500 ms total before users perceive lag. Spend it carefully:")
    BULLET(doc, "Network round-trip and request handling: ~30 ms.")
    BULLET(doc, "Query encoding (BERT pass on GPU): ~10 ms; on CPU: ~50 ms.")
    BULLET(doc, "Domain classification: ~20 ms.")
    BULLET(doc, "FAISS search per domain: ~15 ms (HNSW), ~5 ms (IVF).")
    BULLET(doc, "BM25 search per domain: ~30-100 ms depending on corpus size.")
    BULLET(doc, "Cross-encoder reranking 100 candidates: ~80 ms on GPU, ~2000 ms on CPU.")
    BULLET(doc, "RAG generation (LLM): ~500-2000 ms. Often dominates everything else.")

    H2(doc, "18.2 The three optimisation levers")
    NUM(doc, "Cache aggressively: query embeddings, classification results, retrieval results for popular queries.")
    NUM(doc, "Parallelise per-domain retrieval: BM25 and dense within a domain can run concurrently.")
    NUM(doc, "Quantise: int8 weights for the encoder, int8 vectors for the index. Usually 2-4× speedup with <1% accuracy loss.")

    H2(doc, "18.3 Monitoring")
    P(doc,
      "Three things you must measure in production:")
    BULLET(doc, "Latency percentiles: P50, P95, P99. Tail latency is what users actually feel.")
    BULLET(doc, "Quality drift: re-run a fixed evaluation set weekly. Flag when nDCG drops by more than 0.01.")
    BULLET(doc,
           "Routing distribution: how many queries go to each domain. A sudden shift means something changed in the user population — investigate before tuning.")

    H2(doc, "18.4 Index freshness")
    P(doc,
      "FAISS indexes are static — adding a vector requires rebuilding the index. For dynamic corpora, "
      "use a tiered architecture: a large 'cold' index rebuilt nightly, plus a small 'hot' in-memory index "
      "that holds the day's new documents. Search both, fuse the results.")

    H2(doc, "18.5 The reliability vs accuracy trade-off")
    P(doc,
      "A search system that returns slightly worse results 100% of the time is better than one that returns "
      "perfect results 95% of the time and 500 errors the other 5%. Always have a fallback path: if reranking "
      "fails, return the bi-encoder results. If the dense retriever fails, return BM25. The user should never "
      "see an error page.")

    doc.add_page_break()


def chapter_19(doc):
    H1(doc, "Chapter 19 — Practical Lessons from Building This Project")

    P(doc,
      "This guide accompanies a multi-domain IR search engine project. These are the lessons learned during "
      "implementation that no textbook will tell you.")

    H2(doc, "19.1 Encoder selection dominates everything")
    P(doc,
      "We measured a 30× difference in nDCG@10 between domains using retrieval-trained encoders (BGE) and "
      "domains using non-retrieval MLM encoders (SciBERT, BioBERT, ClinicalBERT) on the same corpus type. "
      "Encoder choice mattered more than corpus size, more than index type, more than fusion strategy. "
      "If you remember one thing from this chapter, remember: use a retrieval-trained encoder.")

    H2(doc, "19.2 BM25 is not optional")
    P(doc,
      "When the dense encoder is weak, BM25 is what saves you. We measured BM25 outperforming dense retrieval "
      "in 4 out of 6 domains because the dense encoders were MLMs. Even when the dense encoder is good, hybrid "
      "retrieval still adds 2-5% nDCG and substantially improves cross-domain coverage. Always enable BM25; "
      "the cost of building the index is trivial compared to the quality gain.")

    H2(doc, "19.3 Routing is brittle without real training data")
    P(doc,
      "Two of our six domains had no real training queries — only synthetic seed queries. Their routing accuracy "
      "was 26% and 40%, compared to 95-98% for domains with thousands of real queries. The classifier cannot "
      "learn what real users sound like from synthetic data alone. Always invest in collecting real query examples.")

    H2(doc, "19.4 Evaluation bugs hide for months")
    P(doc,
      "We discovered two critical bugs that had been silently destroying our metrics for weeks: a corpus loader "
      "that captured only 0.2% of relevant documents, and a classifier evaluation that was testing on training data. "
      "Both produced numbers that looked plausible. The lesson: when an evaluation result is much better or much "
      "worse than expected, do not celebrate or despair — look for the bug first.")

    H2(doc, "19.5 The 80/20 of latency")
    P(doc,
      "In our system, retrieval (FAISS + BM25) accounted for 77% of latency, classification 23%, and "
      "fusion <1%. Optimising fusion was a waste of time; optimising the retrieval path was worth weeks. "
      "Measure first, optimise second.")

    H2(doc, "19.6 The cost of correct ablation")
    P(doc,
      "A full ablation (3 retrieval conditions × 6 domains × 50 queries) took 70 minutes on CPU. "
      "Doing it on the full query set would take 10+ hours. Use sampling for ablations; reserve the full evaluation "
      "for final results. Always compare conditions on the SAME sample, not on different samples — that is "
      "the only way to control for query difficulty variance.")

    KEY(doc,
        "The hardest part of IR engineering is not the algorithms. It is keeping the evaluation honest, "
        "the data clean, and the system observable enough to know when something has gone wrong.")

    doc.add_page_break()


def chapter_20(doc):
    H1(doc, "Chapter 20 — Open Problems and What to Read Next")

    H2(doc, "20.1 Open problems")
    BULLET(doc, "Long-context retrieval: how to retrieve over book-length documents without losing fine-grained relevance.")
    BULLET(doc, "Multi-modal retrieval: text, image, audio, code in a single index.")
    BULLET(doc, "Personalised retrieval: incorporating user history without catastrophic privacy leaks.")
    BULLET(doc, "Faithful RAG: forcing LLMs to use only the retrieved evidence, not their parametric memory.")
    BULLET(doc, "Continuous learning: updating dense indexes incrementally without rebuilding.")

    H2(doc, "20.2 Foundational papers (read in order)")
    NUM(doc, "Salton & Buckley (1988) — Term weighting approaches in automatic text retrieval. The TF-IDF paper.")
    NUM(doc, "Robertson & Walker (1994) — Some simple effective approximations to the 2-Poisson model. The BM25 paper.")
    NUM(doc, "Mikolov et al. (2013) — Efficient Estimation of Word Representations in Vector Space. Word2Vec.")
    NUM(doc, "Vaswani et al. (2017) — Attention Is All You Need. The Transformer paper.")
    NUM(doc, "Devlin et al. (2018) — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.")
    NUM(doc, "Reimers & Gurevych (2019) — Sentence-BERT. Bi-encoders that actually work.")
    NUM(doc, "Karpukhin et al. (2020) — Dense Passage Retrieval for Open-Domain Question Answering. DPR.")
    NUM(doc, "Khattab & Zaharia (2020) — ColBERT: Efficient and Effective Passage Search via Contextualised Late Interaction.")
    NUM(doc, "Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. The RAG paper.")
    NUM(doc, "Thakur et al. (2021) — BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.")
    NUM(doc, "Santhanam et al. (2022) — ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.")

    H2(doc, "20.3 Tools to learn deeply")
    BULLET(doc, "Lucene / OpenSearch / Elasticsearch — the production-grade inverted index ecosystem.")
    BULLET(doc, "FAISS — Facebook's vector search library, the standard.")
    BULLET(doc, "sentence-transformers — the easiest way to use bi-encoders.")
    BULLET(doc, "ir_measures — the standard library for computing retrieval metrics.")
    BULLET(doc, "BEIR — the standard out-of-domain evaluation framework.")
    BULLET(doc, "pyserini — Lucene + dense retrieval for research.")
    BULLET(doc, "RAGAS — RAG-specific evaluation.")

    H2(doc, "20.4 Final advice")
    P(doc,
      "Build a small system end-to-end before reading more papers. Index 10,000 documents, write a query, "
      "watch the results come back. Then change one thing — swap the encoder, enable BM25, change the chunking — "
      "and measure the difference. Every paper you read after that will be grounded in something concrete.")
    P(doc,
      "IR is not a field where you can think your way to the right answer. It is a field where you build a "
      "small thing, measure it, change it, measure again, and slowly accumulate intuition about what works "
      "in your data. The papers and the formulas tell you which knobs exist; only your data tells you which "
      "ones to turn.")

    KEY(doc,
        "Theory tells you what to try. Evaluation tells you what works. Build the loop; the loop teaches you the field.")

    doc.add_page_break()


def appendix(doc):
    H1(doc, "Appendix — A One-Page IR Cheat Sheet")

    H2(doc, "Lexical retrieval")
    BULLET(doc, "BM25 score = Σ idf(t) · saturating-tf(t, d, k1, b)")
    BULLET(doc, "Default k1 = 1.2, b = 0.75. Tune only after measuring.")
    BULLET(doc, "Inverted index + WAND/MaxScore = milliseconds per query.")

    H2(doc, "Dense retrieval")
    BULLET(doc, "Bi-encoder + FAISS HNSW for retrieval.")
    BULLET(doc, "Use a retrieval-trained encoder (BGE, E5, GTE), not an MLM (SciBERT, BioBERT).")
    BULLET(doc, "L2-normalise vectors so cosine = dot product.")
    BULLET(doc, "HNSW M=32, efConstruction=200, efSearch=128 is a strong default.")

    H2(doc, "Hybrid")
    BULLET(doc, "Always combine BM25 and dense with RRF (k=60).")
    BULLET(doc, "Hybrid > Dense for non-retrieval encoders, ≈ Dense for retrieval encoders, never worse.")

    H2(doc, "Reranking")
    BULLET(doc, "Cross-encoder rerank top 100 → top 10 if you have GPU.")
    BULLET(doc, "ms-marco-MiniLM-L-6-v2 is the fastest competent reranker.")

    H2(doc, "Evaluation")
    BULLET(doc, "Use nDCG@10 as the default metric.")
    BULLET(doc, "Report on the full BEIR-style heldout set, not on training-distribution queries.")
    BULLET(doc, "Always control for query difficulty by comparing methods on identical query samples.")

    H2(doc, "RAG")
    BULLET(doc, "Chunk size 200-500 tokens, overlap 10-20%.")
    BULLET(doc, "Retrieve 20-50 candidates, rerank to 5-10, send to LLM.")
    BULLET(doc, "Measure faithfulness AND answer relevance.")

    H2(doc, "Production")
    BULLET(doc, "Latency budget 500 ms. Cache aggressively.")
    BULLET(doc, "Always have a fallback path. Never return errors.")
    BULLET(doc, "Monitor latency P95, quality drift, and routing distribution.")

    H2(doc, "The unwritten rule")
    BULLET(doc, "Encoder selection dominates everything. If you only optimise one thing, optimise that.")


def build(doc):
    title_page(doc)
    how_to_read(doc)

    # Table of contents (manual since we don't auto-update fields)
    H1(doc, "Table of Contents")
    toc = [
        "Part I — Foundations",
        "  Ch 1: What Is Information Retrieval, Really?",
        "  Ch 2: Text Preprocessing — Making Text Comparable",
        "Part II — Classical Lexical IR",
        "  Ch 3: Term Frequency — The First Honest Attempt",
        "  Ch 4: Inverse Document Frequency — Why and How",
        "  Ch 5: Vector Space Model and Cosine Similarity",
        "  Ch 6: BM25 — Why TF-IDF Wasn't Enough",
        "  Ch 7: Inverted Indexes — Making Search Scalable",
        "Part III — Neural IR",
        "  Ch 8: Word Embeddings — From Symbols to Geometry",
        "  Ch 9: Transformers and Attention from First Principles",
        "  Ch 10: BERT and Contextual Embeddings",
        "  Ch 11: Bi-Encoders, Cross-Encoders, Sentence-BERT",
        "  Ch 12: ANN Search and FAISS",
        "Part IV — Hybrid, Ranking, Evaluation",
        "  Ch 13: Hybrid Retrieval and Reciprocal Rank Fusion",
        "  Ch 14: Rerankers and Learning to Rank",
        "  Ch 15: Evaluation Metrics — What 'Good' Means",
        "Part V — RAG and Systems",
        "  Ch 16: Retrieval-Augmented Generation",
        "  Ch 17: Query Routing and Multi-Domain Systems",
        "  Ch 18: Production Concerns",
        "  Ch 19: Practical Lessons from Building This Project",
        "  Ch 20: Open Problems and What to Read Next",
        "Appendix — One-Page IR Cheat Sheet",
    ]
    for line in toc:
        p = doc.add_paragraph(line)
        p.paragraph_format.space_after = Pt(2)
    doc.add_page_break()

    chapter_1(doc)
    chapter_2(doc)
    chapter_3(doc)
    chapter_4(doc)
    chapter_5(doc)
    chapter_6(doc)
    chapter_7(doc)
    chapter_8(doc)
    chapter_9(doc)
    chapter_10(doc)
    chapter_11(doc)
    chapter_12(doc)
    chapter_13(doc)
    chapter_14(doc)
    chapter_15(doc)
    chapter_16(doc)
    chapter_17(doc)
    chapter_18(doc)
    chapter_19(doc)
    chapter_20(doc)
    appendix(doc)


if __name__ == "__main__":
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    build(doc)
    doc.save("results/IR_LEARNING_GUIDE.docx")
    print("Saved: results/IR_LEARNING_GUIDE.docx")
