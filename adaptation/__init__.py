"""
adaptation/__init__.py
-----------------------
Exports the public interface of the adaptation package.

Other parts of the project should import from here:

    from adaptation import QueryTopicLearner, QueryFeedback, TOPICS
    from adaptation import HybridWeightAdapter, FeedbackEvent
    from adaptation import plot_prequential

Do NOT import directly from submodules outside this package.
"""

from adaptation.online_learner import (
    QueryTopicLearner,
    QueryFeedback,
    FeedbackEvent,
    HybridWeightAdapter,
    LearnerState,
    TOPICS,
)
from adaptation.plot_metrics import plot_prequential

__all__ = [
    "QueryTopicLearner",     # online learner — used by the agent (D3)
    "QueryFeedback",         # input dataclass — used by /feedback endpoint (D2)
    "FeedbackEvent",         # feedback event — used by HybridWeightAdapter
    "HybridWeightAdapter",   # fusion weight adapter — used by retrieval (D2)
    "LearnerState",          # snapshot dataclass — used by plot_metrics
    "TOPICS",                # topic list — used by Neo4j seed script (D2)
    "plot_prequential",      # chart generator — used by D1 notebook
]