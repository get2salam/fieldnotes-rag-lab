"""Deterministic evaluation harness for the RAG pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .retriever import Retriever
from .synthesis import Synthesizer


@dataclass
class EvalQuery:
    """A single evaluation query with ground-truth expectations."""

    query_id: str
    query: str
    expected_doc_ids: List[str]       # Documents that must appear in results
    expected_keywords: List[str]       # Words that should appear in the answer
    min_citations: int = 1


@dataclass
class EvalResult:
    """Result for a single evaluated query."""

    query_id: str
    query: str
    recall_at_k: float
    mrr: float
    citation_accuracy: float
    answer_coverage: float
    retrieved_doc_ids: List[str]
    answer: str

    @property
    def composite_score(self) -> float:
        """Weighted average of all metrics."""
        return (
            0.35 * self.recall_at_k
            + 0.25 * self.mrr
            + 0.20 * self.citation_accuracy
            + 0.20 * self.answer_coverage
        )


@dataclass
class EvalReport:
    """Aggregate evaluation report."""

    results: List[EvalResult] = field(default_factory=list)

    @property
    def mean_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.recall_at_k for r in self.results) / len(self.results)

    @property
    def mean_mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.mrr for r in self.results) / len(self.results)

    @property
    def mean_citation_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.citation_accuracy for r in self.results) / len(self.results)

    @property
    def mean_answer_coverage(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.answer_coverage for r in self.results) / len(self.results)

    @property
    def mean_composite(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.composite_score for r in self.results) / len(self.results)

    def format(self) -> str:
        lines = [
            "=" * 60,
            "FieldNotes RAG Lab — Evaluation Report",
            "=" * 60,
            f"Queries evaluated: {len(self.results)}",
            "",
            f"Recall@k:           {self.mean_recall:.3f}",
            f"MRR:                {self.mean_mrr:.3f}",
            f"Citation accuracy:  {self.mean_citation_accuracy:.3f}",
            f"Answer coverage:    {self.mean_answer_coverage:.3f}",
            f"Composite score:    {self.mean_composite:.3f}",
            "",
            "Per-query breakdown:",
        ]
        for r in self.results:
            lines.append(
                f"  [{r.query_id}] recall={r.recall_at_k:.2f} "
                f"mrr={r.mrr:.2f} "
                f"coverage={r.answer_coverage:.2f} "
                f"composite={r.composite_score:.2f}"
            )
            lines.append(f"    Q: {r.query[:70]}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "num_queries": len(self.results),
                "mean_recall_at_k": round(self.mean_recall, 4),
                "mean_mrr": round(self.mean_mrr, 4),
                "mean_citation_accuracy": round(self.mean_citation_accuracy, 4),
                "mean_answer_coverage": round(self.mean_answer_coverage, 4),
                "mean_composite": round(self.mean_composite, 4),
            },
            "per_query": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "recall_at_k": round(r.recall_at_k, 4),
                    "mrr": round(r.mrr, 4),
                    "citation_accuracy": round(r.citation_accuracy, 4),
                    "answer_coverage": round(r.answer_coverage, 4),
                    "composite_score": round(r.composite_score, 4),
                    "retrieved_doc_ids": r.retrieved_doc_ids,
                    "answer_preview": r.answer[:200],
                }
                for r in self.results
            ],
        }


def _recall_at_k(
    expected: List[str], retrieved_ids: List[str]
) -> float:
    if not expected:
        return 1.0
    found = sum(1 for e in expected if e in retrieved_ids)
    return found / len(expected)


def _mrr(expected: List[str], retrieved_ids: List[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def _answer_coverage(answer: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return found / len(keywords)


class Evaluator:
    """Run a suite of evaluation queries against the RAG pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        synthesizer: Synthesizer,
    ) -> None:
        self.retriever = retriever
        self.synthesizer = synthesizer

    def evaluate_query(self, eq: EvalQuery) -> EvalResult:
        chunks, citations = self.retriever.retrieve_with_citations(
            eq.query, top_k=self.retriever.top_k
        )
        result = self.synthesizer.synthesize(eq.query, chunks)

        retrieved_doc_ids = [c.doc_id for c, _ in chunks]

        recall = _recall_at_k(eq.expected_doc_ids, retrieved_doc_ids)
        mrr = _mrr(eq.expected_doc_ids, retrieved_doc_ids)
        cit_acc = 1.0 if len(citations) >= eq.min_citations else 0.0
        coverage = _answer_coverage(result.answer, eq.expected_keywords)

        return EvalResult(
            query_id=eq.query_id,
            query=eq.query,
            recall_at_k=recall,
            mrr=mrr,
            citation_accuracy=cit_acc,
            answer_coverage=coverage,
            retrieved_doc_ids=retrieved_doc_ids,
            answer=result.answer,
        )

    def run(self, queries: List[EvalQuery]) -> EvalReport:
        report = EvalReport()
        for eq in queries:
            result = self.evaluate_query(eq)
            report.results.append(result)
        return report

    @staticmethod
    def load_fixtures(path: str | Path) -> List[EvalQuery]:
        """Load evaluation fixtures from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [
            EvalQuery(
                query_id=item["query_id"],
                query=item["query"],
                expected_doc_ids=item.get("expected_doc_ids", []),
                expected_keywords=item.get("expected_keywords", []),
                min_citations=item.get("min_citations", 1),
            )
            for item in data
        ]
