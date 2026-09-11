"""Command-line entry points for TNFR spectral factorization."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from tnfr.dynamics.advanced_fft_arithmetic import TNFRAdvancedFFTEngine
from tnfr.dynamics.distributed_fft import DistributedFFTEngine, annotate_dispatcher
from tnfr.dynamics.fft_backend import FFTBackend
from tnfr.dynamics.fft_dispatchers import HTTPFFTDispatcher, ThreadedQueueDispatcher

from .spectral_paley import (
    SpectralAnalysisResult,
    SpectralPaleyFactorizer,
    _load_callable_dispatcher,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run TNFR Paley-spectrum factorization using SpectralPaleyFactorizer "
            "and emit coherence telemetry for each query."
        )
    )
    parser.add_argument(
        "targets", nargs="+", type=int, help="Composite candidates to analyze."
    )
    parser.add_argument(
        "--modulus",
        type=int,
        default=None,
        help="Override the Paley modulus (must be odd and 1 mod 4).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Maximum Paley modulus allowed (defaults to 4,097 when omitted; use 0 for no cap).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON (one object per target) instead of a text summary.",
    )
    parser.add_argument(
        "--fft-backend",
        choices=("local", "distributed"),
        default="local",
        help="Select the FFT backend implementation. Use 'distributed' to enable the queue dispatcher.",
    )
    parser.add_argument(
        "--fft-dispatcher",
        default="queue",
        help=(
            "Dispatcher to use when --fft-backend=distributed. Use 'queue' for the local threaded"
            " dispatcher, provide an HTTP(S) endpoint, or specify local:package.module:callable"
            " to import a custom dispatcher factory."
        ),
    )
    parser.add_argument(
        "--dispatcher-workers",
        type=int,
        default=1,
        help="Maximum worker threads for the queue dispatcher (distributed backend only).",
    )
    parser.add_argument(
        "--dispatcher-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for queue dispatcher responses (distributed backend only).",
    )
    parser.add_argument(
        "--dispatcher-serializer",
        choices=("identity", "pickle"),
        default="identity",
        help="Payload codec for queue dispatcher requests/responses.",
    )
    parser.add_argument(
        "--dispatcher-http-token",
        default=None,
        help="Optional bearer token for HTTP dispatchers (distributed backend only).",
    )
    return parser


def _result_payload(result: SpectralAnalysisResult) -> Mapping[str, object]:
    terms = result.arithmetic_terms
    if is_dataclass(terms):
        terms_payload = asdict(terms)  # type: ignore[arg-type]
    else:
        terms_payload = {
            "tau": getattr(terms, "tau", None),
            "sigma": getattr(terms, "sigma", None),
            "omega": getattr(terms, "omega", None),
        }

    return {
        "n": result.n,
        "modulus": result.modulus,
        "node_count": result.node_count,
        "edge_count": result.edge_count,
        "laplacian_gap": result.laplacian_gap,
        "candidate_factors": result.candidate_factors,
        "coherence_score": result.coherence_score,
        "phi_s": result.phi_s,
        "phase_gradient": result.phase_gradient,
        "phase_curvature": result.phase_curvature,
        "coherence_length": result.coherence_length,
        "arithmetic_delta_nfr": result.arithmetic_delta_nfr,
        "arithmetic_terms": terms_payload,
        "arithmetic_components": result.arithmetic_components,
        "arithmetic_local_coherence": result.arithmetic_local_coherence,
        "notes": result.notes,
        "partition_artifact_dir": result.partition_artifact_dir,
        "partition_manifest_path": result.partition_manifest_path,
        "partition_manifest_index_path": result.partition_manifest_index_path,
        "partition_file_archive_path": result.partition_file_archive_path,
        "dispatcher_telemetry": result.dispatcher_telemetry,
        "tnfr_certified_factors": result.tnfr_certified_factors,
        "tnfr_factor_signature": result.tnfr_factor_signature,
    }


def _format_mapping(mapping: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for key in sorted(mapping):
        value = mapping[key]
        if isinstance(value, Mapping):
            value_str = json.dumps(value, sort_keys=True)
        else:
            value_str = str(value)
        parts.append(f"{key}={value_str}")
    return ", ".join(parts)


def _format_result(result: SpectralAnalysisResult) -> Iterable[str]:
    yield f"Spectral TNFR factorization for n={result.n}"
    yield f"  modulus={result.modulus} (nodes={result.node_count}, edges={result.edge_count})"
    yield f"  laplacian_gap={result.laplacian_gap:.6f} | coherence_score={result.coherence_score:.6f}"
    yield (
        "  phi_s={:.4f}, grad_phi={:.6f}, k_phi={:.6f}, xi_c={:.4f}".format(
            result.phi_s,
            result.phase_gradient,
            result.phase_curvature,
            result.coherence_length,
        )
    )
    yield (
        f"  delta_nfr={result.arithmetic_delta_nfr:.6e}, "
        f"local_coherence={result.arithmetic_local_coherence:.6f}"
    )
    factors = result.candidate_factors or ["-"]
    yield f"  candidate_factors={','.join(str(f) for f in factors)}"
    if result.tnfr_certified_factors:
        yield f"  tnfr_certified={','.join(str(f) for f in result.tnfr_certified_factors)}"
    if result.tnfr_factor_signature:
        yield f"  tnfr_signature={result.tnfr_factor_signature.get('hash')}"
    if result.fft_backend:
        yield f"  fft_backend={result.fft_backend}"
    if result.dispatcher_telemetry:
        yield f"  dispatcher={_format_mapping(result.dispatcher_telemetry)}"
    if result.partition_artifact_dir:
        yield f"  partition_artifacts={result.partition_artifact_dir}"
    if result.partition_manifest_path:
        yield f"  partition_manifest={result.partition_manifest_path}"
    if result.partition_manifest_index_path:
        yield f"  partition_manifest_summary={result.partition_manifest_index_path}"
    if result.partition_file_archive_path:
        yield f"  partition_file_archive={result.partition_file_archive_path}"
    if result.notes:
        yield f"  notes={result.notes}"


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    max_nodes = args.max_nodes
    if max_nodes == 0:
        max_nodes = None
    fft_backend = _select_fft_backend_from_cli(args)
    factorizer = SpectralPaleyFactorizer(max_nodes=max_nodes, fft_engine=fft_backend)

    results: List[SpectralAnalysisResult] = []
    for target in args.targets:
        result = factorizer.analyze(target, modulus=args.modulus)
        results.append(result)

    if args.json:
        payload = [_result_payload(result) for result in results]
        print(json.dumps(payload, indent=2, default=float))
    else:
        for idx, result in enumerate(results):
            for line in _format_result(result):
                print(line)
            if idx < len(results) - 1:
                print()

    return 0


def _select_fft_backend_from_cli(args: argparse.Namespace) -> Optional[FFTBackend]:
    if args.fft_backend == "local":
        return TNFRAdvancedFFTEngine()
    if args.fft_backend == "distributed":
        dispatcher = _build_dispatcher_from_args(args)
        return DistributedFFTEngine(dispatcher=dispatcher)
    return None


def _build_dispatcher_from_args(
    args: argparse.Namespace,
) -> Callable[[str, Dict[str, Any]], Any]:
    dispatcher_name = (getattr(args, "fft_dispatcher", "queue") or "queue").strip()

    if dispatcher_name.lower() == "queue":
        request_serializer, request_deserializer = _select_codec(
            args.dispatcher_serializer
        )

        dispatcher = ThreadedQueueDispatcher(
            timeout=float(args.dispatcher_timeout),
            max_workers=max(1, int(args.dispatcher_workers or 1)),
            request_serializer=request_serializer,
            request_deserializer=request_deserializer,
            response_serializer=request_serializer,
            response_deserializer=request_deserializer,
        )
        annotate_dispatcher(
            dispatcher.dispatch,
            {
                "type": "queue",
                "source": "cli",
                "max_workers": max(1, int(args.dispatcher_workers or 1)),
                "serializer": args.dispatcher_serializer,
                "timeout": float(args.dispatcher_timeout),
            },
        )
        return dispatcher.dispatch

    if dispatcher_name.startswith("http://") or dispatcher_name.startswith("https://"):
        client = HTTPFFTDispatcher(
            base_url=dispatcher_name,
            auth_token=getattr(args, "dispatcher_http_token", None),
        )
        annotate_dispatcher(
            client.dispatch,
            {
                "type": "http",
                "source": "cli",
                "base_url": dispatcher_name,
                "token_provided": bool(getattr(args, "dispatcher_http_token", None)),
            },
        )
        return client.dispatch

    if dispatcher_name.lower().startswith("local:"):
        spec = dispatcher_name.split(":", 1)[1].strip()
        if not spec:
            raise ValueError("Expected module:function after 'local:' prefix")
        return _build_callable_dispatcher_from_spec(spec, wrapper="local")

    if ":" in dispatcher_name:
        return _build_callable_dispatcher_from_spec(dispatcher_name)

    raise ValueError(
        "Unsupported dispatcher. Use 'queue', an HTTP(S) endpoint, or local:package.module:factory"
    )


def _select_codec(
    name: str,
) -> Tuple[Callable[[Dict[str, Any]], Any], Callable[[Any], Dict[str, Any]]]:
    if name == "identity":
        return (lambda payload: payload, lambda payload: payload)  # type: ignore[return-value]
    if name == "pickle":
        import pickle

        return (
            lambda payload: pickle.dumps(payload),
            lambda blob: pickle.loads(blob),
        )
    raise ValueError(f"Unsupported dispatcher serializer '{name}'")


def _build_callable_dispatcher_from_spec(
    spec: str, *, wrapper: str | None = None
) -> Callable[[str, Dict[str, Any]], Any]:
    metadata = {
        "type": "callable",
        "source": "cli",
        "target": spec,
    }
    if wrapper:
        metadata["wrapper"] = wrapper
    dispatcher = _load_callable_dispatcher(spec, metadata=metadata)
    if dispatcher is None:
        raise ValueError(f"Unable to import dispatcher '{spec}'")
    return dispatcher


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
