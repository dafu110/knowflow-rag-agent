from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .agent import RagAgent
from .chunking import load_documents_from_path
from .evaluation import evaluate
from .models import Principal
from .server import run_server
from .store_factory import create_store


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="KnowFlow Enterprise RAG Agent")
    parser.add_argument("--store", default="data/knowledge_store", help="Knowledge store directory")
    parser.add_argument("--store-backend", choices=["jsonl", "sqlite"], default="jsonl", help="Persistence backend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest files or a directory")
    ingest_parser.add_argument("path")
    ingest_parser.add_argument("--reset", action="store_true")

    ask_parser = subparsers.add_parser("ask", help="Ask a knowledge-grounded question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--user", default="anonymous")
    ask_parser.add_argument("--roles", default="")
    ask_parser.add_argument("--session-id", default="cli")
    ask_parser.add_argument("--top-k", type=int, default=6)

    eval_parser = subparsers.add_parser("eval", help="Run an offline RAG evaluation set")
    eval_parser.add_argument("path")

    serve_parser = subparsers.add_parser("serve", help="Run the local web app")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()
    store = create_store(args.store, backend=args.store_backend)

    if args.command == "ingest":
        if args.reset:
            store.reset()
        documents = load_documents_from_path(Path(args.path))
        added = store.add_documents(documents)
        print(json.dumps({"loaded": len(documents), "added": added, "stats": store.stats()}, ensure_ascii=False, indent=2))
    elif args.command == "ask":
        agent = RagAgent(store)
        principal = Principal(user=args.user, roles=_roles(args.roles))
        answer = agent.ask(args.question, principal=principal, session_id=args.session_id, top_k=args.top_k)
        print(json.dumps(asdict(answer), ensure_ascii=False, indent=2))
    elif args.command == "eval":
        agent = RagAgent(store)
        result = evaluate(agent, Path(args.path))
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    elif args.command == "serve":
        run_server(store=store, host=args.host, port=args.port)


def _roles(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
