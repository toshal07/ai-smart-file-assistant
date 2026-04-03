"""
LangChain Q&A System Module
===========================
LangChain-powered RAG pipeline that reuses the project's retrieval layer
(QueryProcessor + ContextBuilder) and swaps only the answer generation stage
with a LangChain LCEL chain.

Public interface intentionally mirrors QASystem so api_server can switch
between implementations via an environment flag.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional

from src.modules.openai_handler import OpenAIHandler
from src.modules.query_processor import QueryProcessor
from src.modules.context_builder import ContextBuilder
from src.modules.vector_db_setup import VectorDB

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert AI assistant specializing in enterprise \
technology standards, cybersecurity frameworks, and digital governance. Your role \
is to provide comprehensive, well-explained answers based ONLY on the provided document context.

Guidelines:
- Explain concepts clearly in a structured, professional format.
- Use only the provided context. Do not include outside knowledge.
- Cite source document and page when referencing specific facts.
- If context is insufficient, clearly state that limitation.
"""

_HUMAN_PROMPT = """DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer using only the context above and include source citations where possible."""


class LangChainQASystem:
    """
    LangChain-based question answering system.

    Retrieval stays in the existing project modules for consistency.
    Generation runs through a LangChain chain:
      ChatPromptTemplate -> ChatOpenAI -> StrOutputParser
    """

    def __init__(
        self,
        vector_db: VectorDB,
        openai_handler: Optional[OpenAIHandler] = None,
        model: Optional[str] = None,
        top_k: int = 5,
        max_context_tokens: int = 3000,
        min_search_score: float = 0.3,
        temperature: float = 0.3,
        max_tokens: int = 600,
    ):
        self.vector_db = vector_db
        self.openai_handler = openai_handler or OpenAIHandler()
        self.model = model or os.getenv("OPENAI_MODEL", self.openai_handler.model)
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.query_processor = QueryProcessor(
            vector_db=vector_db,
            openai_handler=self.openai_handler,
            default_top_k=top_k,
            min_score=min_search_score,
        )
        self.context_builder = ContextBuilder(
            max_tokens=max_context_tokens,
            min_score=min_search_score,
            include_scores=True,
            deduplicate=True,
        )

        self.conversation_history: List[Dict[str, str]] = []
        self.session_stats = {
            "questions_answered": 0,
            "total_time": 0.0,
            "avg_confidence": 0.0,
            "total_sources_cited": 0,
        }

        self._chain_model = None
        self._chain = None
        self._ensure_chain()

    def _ensure_chain(self):
        """Build or rebuild the chain when model changes at runtime."""
        if self._chain is not None and self._chain_model == self.model:
            return

        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "LangChain OpenAI integration is missing. "
                "Install dependency: pip install langchain-openai"
            ) from exc

        llm_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": self.openai_handler.api_key,
        }
        if self.openai_handler.base_url:
            llm_kwargs["base_url"] = self.openai_handler.base_url

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", _HUMAN_PROMPT),
            ]
        )
        llm = ChatOpenAI(**llm_kwargs)
        self._chain = prompt | llm | StrOutputParser()
        self._chain_model = self.model
        logger.info("LangChain chain initialised with model=%s", self.model)

    def _init_result(self, question: str) -> Dict[str, Any]:
        return {
            "question": question,
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            "search_results": [],
            "context": "",
            "model": self.model or "default",
            "time_seconds": 0.0,
            "error": None,
        }

    def _merge_conversation_into_context(
        self,
        context_str: str,
        use_conversation_history: bool,
    ) -> str:
        if not use_conversation_history or not self.conversation_history:
            return context_str

        turns = self.conversation_history[-6:]
        history_lines: List[str] = []
        for i in range(0, len(turns), 2):
            if i + 1 < len(turns):
                history_lines.append(f"Previous Q: {turns[i]['content']}")
                history_lines.append(f"Previous A: {turns[i + 1]['content']}")
        if not history_lines:
            return context_str

        return "\n".join(history_lines) + "\n\nCurrent Context:\n" + context_str

    def answer_question(
        self,
        question: str,
        filter_source: Optional[str] = None,
        use_conversation_history: bool = False,
    ) -> Dict[str, Any]:
        result = self._init_result(question)

        if not question or not question.strip():
            result["error"] = "Empty question"
            return result

        start = time.time()

        try:
            self._ensure_chain()

            logger.info("LangChain Q&A: '%s'", question[:60])
            search_results = self.query_processor.search_documents(
                question,
                filter_source=filter_source,
            )
            result["search_results"] = search_results

            if not search_results:
                result["answer"] = (
                    "I could not find relevant information in the "
                    "document database to answer your question. "
                    "Please ensure PDFs have been ingested first."
                )
                result["error"] = "no_results"
                result["time_seconds"] = round(time.time() - start, 2)
                return result

            ctx_meta = self.context_builder.build_context_with_metadata(
                search_results,
                question,
            )
            context_str = ctx_meta["context"]
            result["context"] = context_str

            if not context_str:
                result["answer"] = (
                    "The retrieved document chunks did not meet the "
                    "relevance threshold to build a meaningful context. "
                    "Try rephrasing your question."
                )
                result["error"] = "low_relevance"
                result["time_seconds"] = round(time.time() - start, 2)
                return result

            context_str = self._merge_conversation_into_context(
                context_str,
                use_conversation_history,
            )

            answer_text = self._chain.invoke(
                {
                    "question": question,
                    "context": context_str,
                }
            )

            sources = self.context_builder.get_sources_summary(search_results)
            confidence = self._compute_confidence(search_results, answer_text)

            result["answer"] = answer_text
            result["sources"] = sources
            result["confidence"] = confidence
            result["time_seconds"] = round(time.time() - start, 2)

            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer_text})

            self._update_stats(result["time_seconds"], confidence, len(sources))

        except Exception as e:
            result["error"] = str(e)
            result["time_seconds"] = round(time.time() - start, 2)
            logger.error("LangChain Q&A failed: %s", e)

        return result

    def answer_question_stream(
        self,
        question: str,
        filter_source: Optional[str] = None,
        use_conversation_history: bool = False,
    ):
        result = self._init_result(question)

        if not question or not question.strip():
            result["error"] = "Empty question"
            yield result
            return

        start = time.time()

        try:
            self._ensure_chain()

            logger.info("LangChain Q&A Stream: '%s'", question[:60])
            search_results = self.query_processor.search_documents(
                question,
                filter_source=filter_source,
            )
            result["search_results"] = search_results

            if not search_results:
                result["answer"] = (
                    "I could not find relevant information in the "
                    "document database to answer your question. "
                    "Please ensure PDFs have been ingested first."
                )
                result["error"] = "no_results"
                result["time_seconds"] = round(time.time() - start, 2)
                yield result
                yield result["answer"]
                return

            ctx_meta = self.context_builder.build_context_with_metadata(
                search_results,
                question,
            )
            context_str = ctx_meta["context"]
            result["context"] = context_str

            if not context_str:
                result["answer"] = (
                    "The retrieved document chunks did not meet the "
                    "relevance threshold to build a meaningful context. "
                    "Try rephrasing your question."
                )
                result["error"] = "low_relevance"
                result["time_seconds"] = round(time.time() - start, 2)
                yield result
                yield result["answer"]
                return

            context_str = self._merge_conversation_into_context(
                context_str,
                use_conversation_history,
            )

            sources = self.context_builder.get_sources_summary(search_results)
            result["sources"] = sources
            result["confidence"] = self._compute_confidence(search_results, "")

            yield result

            answer_text = ""
            for piece in self._chain.stream({"question": question, "context": context_str}):
                chunk = str(piece)
                answer_text += chunk
                yield chunk

            confidence = self._compute_confidence(search_results, answer_text)
            result["answer"] = answer_text
            result["confidence"] = confidence
            result["time_seconds"] = round(time.time() - start, 2)

            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer_text})
            self._update_stats(result["time_seconds"], confidence, len(sources))

            yield result

        except Exception as e:
            result["error"] = str(e)
            result["time_seconds"] = round(time.time() - start, 2)
            logger.error("LangChain Q&A Stream failed: %s", e)
            yield result

    def answer_with_followup(self, question: str) -> Dict[str, Any]:
        return self.answer_question(question, use_conversation_history=True)

    def answer_with_followup_stream(self, question: str):
        return self.answer_question_stream(question, use_conversation_history=True)

    def reset_conversation(self):
        self.conversation_history.clear()
        logger.info("LangChain conversation history cleared")

    def get_session_stats(self) -> Dict[str, Any]:
        stats = self.session_stats.copy()
        n = stats["questions_answered"]
        stats["avg_time_seconds"] = round(stats["total_time"] / n, 2) if n else 0
        return stats

    def _update_stats(self, time_seconds: float, confidence: float, sources_count: int):
        self.session_stats["questions_answered"] += 1
        self.session_stats["total_time"] += time_seconds
        self.session_stats["total_sources_cited"] += sources_count

        n = self.session_stats["questions_answered"]
        prev_avg = self.session_stats["avg_confidence"]
        self.session_stats["avg_confidence"] = round(
            (prev_avg * (n - 1) + confidence) / n, 3
        )

    def _compute_confidence(self, search_results: List[Dict], answer: str) -> float:
        if not search_results:
            return 0.0

        top_scores = [r.get("score", 0) for r in search_results[:3]]
        base_confidence = sum(top_scores) / len(top_scores)

        uncertainty_phrases = [
            "i don't know",
            "i cannot find",
            "not mentioned",
            "no information",
            "unclear",
            "not specified",
            "cannot determine",
            "insufficient",
        ]
        answer_lower = answer.lower()
        penalty = sum(0.1 for p in uncertainty_phrases if p in answer_lower)

        confidence = max(0.0, min(0.99, base_confidence - penalty))
        return round(confidence, 3)
