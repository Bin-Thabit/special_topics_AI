import time
import warnings
import numpy as np
import optuna
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer

from retrieval.dense_retriever import build_dense_index
from retrieval.bm25_retriever import load_chunks, build_bm25_index
from retrieval.hybrid_retriever import hybrid_search
from evaluation.metrics import evaluate_retriever

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

CHUNKS_PATH = "data/sample_chunks.json"
GOLD_PATH   = "data/gold_qa.json"
MODEL_NAME  = "BAAI/bge-small-en"


def get_embeddings(
    chunks: list[dict],
    model: SentenceTransformer,
    normalize: bool,
    svd_dim: int | None
) -> tuple[np.ndarray, TruncatedSVD | None]:
    """
    Encodes chunks then optionally applies SVD + normalization.
    Returns (embeddings, fitted_svd) so queries can be projected
    into the same reduced space at search time.
    """
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    svd = None
    if svd_dim is not None:
        svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        embeddings = svd.fit_transform(embeddings)  # (n, svd_dim)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

    return embeddings, svd


def run_bohb(n_trials: int = 50, min_budget: int = 5, max_budget: int = 22,chunks_path: str = CHUNKS_PATH, gold_path: str = GOLD_PATH) -> dict:
    """
    BOHB = Bayesian Optimization + HyperBand pruning.

    How it works:
      - HyperbandPruner divides trials into brackets/rungs.
        Each rung sees more questions (budget) before deciding
        whether to prune the trial early.
      - TPESampler is the Bayesian part — it learns which regions
        of the search space are promising and samples from there.
      - Together: bad trials are killed early (Hyperband),
        good regions are explored more (TPE). Faster than plain TPE.

    Budget = number of gold questions evaluated before a pruning check.
      min_budget=5  → first rung: score on 5 questions, prune if bad
      max_budget=22 → final rung: score on all questions (we have 22)
    """
    print("Loading chunks and building BM25 index...")
    chunks     = load_chunks(chunks_path)    
    bm25_index = build_bm25_index(chunks)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"\n🔍 Starting BOHB search ({n_trials} trials)...\n")

    # load gold questions once so we can slice them per budget step
    import json
    with open(gold_path) as f:               # ← use parameter
        gold_questions = json.load(f)
    total_questions = len(gold_questions)

    def objective(trial: optuna.Trial) -> float:
        # ── Step 1: Suggest parameters ───────────────────────
        k         = trial.suggest_int("k", 3, 20)
        alpha     = trial.suggest_float("alpha", 0.0, 1.0)
        svd_dim   = trial.suggest_categorical("svd_dim", [None, 64, 128, 256])
        normalize = trial.suggest_categorical("normalize", [True, False])

        # ── Step 2: Build embeddings ──────────────────────────
        try:
            embeddings, svd = get_embeddings(chunks, model, normalize, svd_dim)
        except Exception:
            return 0.0

        # ── Step 3: Build search function ────────────────────
        def search_fn(query: str, k: int) -> list[dict]:
            query_vector = model.encode(query, convert_to_numpy=True)

            if svd is not None:
                query_vector = svd.transform(
                    query_vector.reshape(1, -1)
                )[0]

            if normalize:
                norm = np.linalg.norm(query_vector)
                if norm > 0:
                    query_vector = query_vector / norm

            return hybrid_search(
                query, chunks,
                bm25_index,
                model, embeddings,
                k=k,
                alpha=alpha,
                query_vector=query_vector
            )

        # ── Step 4: Evaluate in budget steps (enables pruning) ─
        # Hyperband prunes after each rung — we simulate this by
        # evaluating on increasing slices of questions and reporting
        # intermediate NDCG scores after each slice.
        #
        # Rung schedule example (min=5, max=22):
        #   step 0 → evaluate on questions  0–4   (5 questions)
        #   step 1 → evaluate on questions  0–10  (11 questions)
        #   step 2 → evaluate on questions  0–16  (17 questions)
        #   step 3 → evaluate on questions  0–21  (22 questions)
        #
        # After each step, trial.report() tells Optuna the current
        # score, and trial.should_prune() checks if Hyperband wants
        # to kill this trial based on the rung cutoff.


        n_steps   = 4  # number of Hyperband rungs
        step_size = max(1, total_questions // n_steps)
        last_ndcg = 0.0

        start = time.time()

        for step in range(n_steps):
            # grow the slice each step; last step uses all questions
            budget = min((step + 1) * step_size, total_questions)
            subset = gold_questions[:budget]

            # write a temp gold file for evaluate_retriever
            import tempfile, json as _json
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                _json.dump(subset, tmp)
                tmp_path = tmp.name

            metrics   = evaluate_retriever(search_fn, tmp_path, k=k)
            last_ndcg = metrics[f"ndcg@{k}"]

            # report intermediate value to Optuna
            trial.report(last_ndcg, step=step)

            # let Hyperband prune this trial if it's underperforming
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        latency = time.time() - start

        trial.set_user_attr("recall",    metrics[f"recall@{k}"])
        trial.set_user_attr("latency_s", round(latency, 3))

        return last_ndcg

    # ── BOHB study: TPE sampler + Hyperband pruner ────────────
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=min_budget,
            max_resource=max_budget,
            reduction_factor=3       # each rung keeps top 1/3 of trials
        )
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── Extract best results ──────────────────────────────────
    best         = study.best_trial
    best_params  = best.params
    best_ndcg    = best.value
    best_recall  = best.user_attrs.get("recall", None)
    best_latency = best.user_attrs.get("latency_s", None)

    pruned  = len([t for t in study.trials
                   if t.state == optuna.trial.TrialState.PRUNED])
    complete = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])

    print(f"\n{'='*50}")
    print(f"✅ BOHB search complete!")
    print(f"{'='*50}")
    print(f"Completed trials : {complete}")
    print(f"Pruned trials    : {pruned}  ← trials killed early by Hyperband")
    print(f"Best NDCG@k      : {best_ndcg:.4f}")
    print(f"Best Recall@k    : {best_recall}")
    print(f"Best Latency     : {best_latency}s")
    print(f"\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param:<12} : {value}")

    return {
        "best_params":  best_params,
        "best_ndcg":    best_ndcg,
        "best_recall":  best_recall,
        "best_latency": best_latency,
        "n_trials":     n_trials,
        "pruned":       pruned,
        "complete":     complete,
        "study":        study,
    }

