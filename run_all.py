"""
run_all.py — 전체 파이프라인을 한 번에 실행

Usage:
    python run_all.py                        # smoke test (기본값, CPU, 빠름)
    python run_all.py --full                 # 전체 데이터 (GPU, 오래 걸림)
    python run_all.py --domain science       # 특정 도메인만
    python run_all.py --skip-corpus          # corpus 빌드 건너뜀 (이미 있으면)
    python run_all.py --skip-index           # 인덱스 빌드 건너뜀
    python run_all.py --skip-train           # 분류기 학습 건너뜀
    python run_all.py --eval-only            # 평가만 실행
    python run_all.py --search "my query"    # 검색 데모만
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ── ANSI 색상 ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def info(msg: str)    -> None: print(f"{CYAN}[INFO]{RESET}  {msg}")
def ok(msg: str)      -> None: print(f"{GREEN}[ OK ]{RESET}  {msg}")
def warn(msg: str)    -> None: print(f"{YELLOW}[WARN]{RESET}  {msg}")
def error(msg: str)   -> None: print(f"{RED}[FAIL]{RESET}  {msg}")
def header(msg: str)  -> None: print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n{BOLD}  {msg}{RESET}\n{'─'*60}")


# ── 단계 실행기 ─────────────────────────────────────────────────────────────
def run_step(
    step_name: str,
    cmd: list[str],
    skip: bool = False,
) -> bool:
    """한 단계를 실행하고 성공 여부를 반환합니다."""
    if skip:
        warn(f"[{step_name}] 건너뜀 (--skip 플래그)")
        return True

    header(step_name)
    info(f"실행: {' '.join(cmd)}")
    t0 = time.perf_counter()

    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        ok(f"[{step_name}] 완료 ({elapsed:.1f}초)")
        return True
    else:
        error(f"[{step_name}] 실패 (exit code {result.returncode})")
        return False


# ── 메인 ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-domain 검색 엔진 전체 파이프라인 실행기"
    )
    parser.add_argument("--full", action="store_true",
                        help="전체 데이터셋 사용 (configs/full.yaml 적용)")
    parser.add_argument("--domain", default=None,
                        help="특정 도메인만 (예: science, medical, legal ...)")
    parser.add_argument("--skip-corpus", action="store_true",
                        help="Step 1 corpus 빌드 건너뜀")
    parser.add_argument("--skip-index", action="store_true",
                        help="Step 2 FAISS 인덱스 빌드 건너뜀")
    parser.add_argument("--skip-bm25", action="store_true",
                        help="Step 3 BM25 인덱스 빌드 건너뜀 (기본값: 건너뜀)")
    parser.add_argument("--with-bm25", action="store_true",
                        help="BM25 인덱스도 빌드 (Java 11+ 필요)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Step 4 분류기 학습 건너뜀")
    parser.add_argument("--eval-only", action="store_true",
                        help="평가만 실행 (corpus/index/train 모두 건너뜀)")
    parser.add_argument("--search", default=None, metavar="QUERY",
                        help="검색 데모 쿼리 (평가 대신 단일 쿼리 실행)")
    parser.add_argument("--ablation", action="store_true",
                        help="평가 시 ablation 조건 모두 실행")
    parser.add_argument("--latency", action="store_true",
                        help="평가 시 레이턴시 측정 포함")
    args = parser.parse_args()

    # ── 설정 파일 결정 ─────────────────────────────────────────────────────
    base_cfg = "configs/default.yaml"
    override_cfg = "configs/full.yaml" if args.full else None

    def make_cmd(script: str, extra: list[str] | None = None) -> list[str]:
        cmd = [sys.executable, f"scripts/{script}", "--config", base_cfg]
        if override_cfg:
            cmd += ["--override", override_cfg]
        if args.domain:
            cmd += ["--domain", args.domain]
        if extra:
            cmd += extra
        return cmd

    total_t0 = time.perf_counter()
    steps_ok = []
    steps_fail = []

    # eval-only 플래그 처리
    if args.eval_only:
        args.skip_corpus = True
        args.skip_index = True
        args.skip_train = True

    print(f"""
{BOLD}{'='*60}
  Multi-Domain 검색 엔진 파이프라인
{'='*60}{RESET}
  Config  : {base_cfg}{f' + {override_cfg}' if override_cfg else ''}
  Domain  : {args.domain or '전체 도메인'}
  Mode    : {'전체 데이터' if args.full else 'Smoke test (소규모)'}
""")

    # ── Step 1: Corpus 빌드 ────────────────────────────────────────────────
    ok1 = run_step(
        "Step 1/5 — Corpus 다운로드 & 청킹",
        make_cmd("build_corpus.py"),
        skip=args.skip_corpus,
    )
    (steps_ok if ok1 else steps_fail).append("corpus")
    if not ok1 and not args.skip_corpus:
        error("corpus 빌드 실패 → 중단")
        sys.exit(1)

    # ── Step 2: FAISS 인덱스 ──────────────────────────────────────────────
    ok2 = run_step(
        "Step 2/5 — FAISS 인덱스 빌드",
        make_cmd("build_faiss.py"),
        skip=args.skip_index,
    )
    (steps_ok if ok2 else steps_fail).append("faiss")
    if not ok2 and not args.skip_index:
        error("FAISS 빌드 실패 → 중단")
        sys.exit(1)

    # ── Step 3: BM25 인덱스 (선택) ────────────────────────────────────────
    # full 모드이거나 --with-bm25 플래그가 있을 때만 실행
    build_bm25 = (args.full or args.with_bm25) and not args.skip_bm25
    ok3 = run_step(
        "Step 3/5 — BM25 인덱스 빌드 (선택)",
        # domain 없이 실행: bm25_enabled 도메인만 자동 선택
        [sys.executable, "scripts/build_bm25.py",
         "--config", base_cfg] + (["--override", override_cfg] if override_cfg else []),
        skip=not build_bm25,
    )
    (steps_ok if ok3 else steps_fail).append("bm25")

    # ── Step 4: 도메인 분류기 학습 ────────────────────────────────────────
    ok4 = run_step(
        "Step 4/5 — 도메인 분류기 학습 (DistilBERT)",
        # domain 플래그 없음: 분류기는 항상 6개 도메인 전체로 학습
        [sys.executable, "scripts/train_domain_classifier.py",
         "--config", base_cfg] + (["--override", override_cfg] if override_cfg else []),
        skip=args.skip_train,
    )
    (steps_ok if ok4 else steps_fail).append("classifier")
    if not ok4 and not args.skip_train:
        error("분류기 학습 실패 → 중단")
        sys.exit(1)

    # ── Step 5: 검색 데모 또는 평가 ──────────────────────────────────────
    if args.search:
        ok5 = run_step(
            "Step 5/5 — 검색 데모",
            [sys.executable, "scripts/search.py",
             "--config", base_cfg,
             "--query", args.search]
            + (["--override", override_cfg] if override_cfg else []),
        )
    else:
        eval_extra = []
        if args.ablation:
            eval_extra.append("--ablation")
        if args.latency:
            eval_extra.append("--latency")
        ok5 = run_step(
            "Step 5/5 — 평가 (nDCG@10, MRR@10, Recall@100)",
            make_cmd("evaluate.py", eval_extra),
        )
    (steps_ok if ok5 else steps_fail).append("eval/search")

    # ── 최종 요약 ─────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_t0
    print(f"""
{BOLD}{'='*60}
  파이프라인 완료  ({total_elapsed:.1f}초)
{'='*60}{RESET}
  ✅  성공: {', '.join(steps_ok) or '없음'}
  ❌  실패: {', '.join(steps_fail) or '없음'}

결과 파일:
  data/processed/{{domain}}/      corpus, queries, qrels parquet
  indexes/faiss/{{domain}}/       faiss.index, docstore, id_mapping
  models/domain_classifier/      학습된 분류기
  results/                       metrics JSON, ablation CSV
""")

    if steps_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
