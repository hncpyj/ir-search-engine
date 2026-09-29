from src.adapters.msmarco_adapter import MSMarcoAdapter
from src.adapters.scifact_adapter import ScifactAdapter
from src.adapters.scidocs_adapter import ScidocsAdapter
from src.adapters.fiqa_adapter import FiqaAdapter
from src.adapters.treccovid_adapter import TrecCovidAdapter
from src.adapters.nfcorpus_adapter import NfcorpusAdapter

ADAPTER_MAP = {
    "general":    MSMarcoAdapter,
    "scidocs":    ScidocsAdapter,
    "science":    ScifactAdapter,
    "finance":    FiqaAdapter,
    "medical":    TrecCovidAdapter,
    "biomedical": NfcorpusAdapter,
}

TOKENIZER_MAP = {
    "general":    "sentence-transformers/msmarco-bert-base-dot-v5",
    "scidocs":    "allenai/scibert_scivocab_uncased",
    "science":    "allenai/scibert_scivocab_uncased",
    "finance":    "ProsusAI/finbert",
    "medical":    "emilyalsentzer/Bio_ClinicalBERT",
    "biomedical": "dmis-lab/biobert-base-cased-v1.1",
}
