from __future__ import annotations
import argparse
import json
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - opcional
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml es opcional
    yaml = None

import networkx as nx

from .constants import inject_defaults, DEFAULTS
from .sense import register_sigma_callback, sigma_series, sigma_rose
from .metrics import (
    register_metrics_callbacks,
    Tg_global,
    latency_series,
    glifogram_series,
    glyph_top,
    export_history,
)
from .trace import register_trace
from .program import play, seq, block, wait, target
from .dynamics import step, _update_history, default_glyph_selector, parametric_glyph_selector, validate_canon
from .gamma import GAMMA_REGISTRY
from .scenarios import build_graph
from .presets import get_preset


def _save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _str2bool(s: str) -> bool:
    s = s.lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def _args_to_dict(args: argparse.Namespace, prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    pref = prefix.replace(".", "_")
    for k, v in vars(args).items():
        if k.startswith(pref) and v is not None:
            out[k[len(pref):]] = v
    return out


def _load_sequence(path: str) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".yaml") or path.endswith(".yml"):
        if not yaml:
            raise RuntimeError("pyyaml no está instalado, usa JSON o instala pyyaml")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    def parse_token(tok: Any):
        if isinstance(tok, str):
            return tok
        if isinstance(tok, dict):
            if "WAIT" in tok:
                return wait(int(tok["WAIT"]))
            if "TARGET" in tok:
                return target(tok["TARGET"])
            if "THOL" in tok:
                spec = tok["THOL"] or {}
                b = [_parse_inner(x) for x in spec.get("body", [])]
                return block(*b, repeat=int(spec.get("repeat", 1)), close=spec.get("close"))
        raise ValueError(f"Token inválido: {tok}")

    def _parse_inner(x: Any):
        return parse_token(x)

    return [parse_token(t) for t in data]


def _attach_callbacks(G: nx.Graph) -> None:
    inject_defaults(G, DEFAULTS)
    register_sigma_callback(G)
    register_metrics_callbacks(G)
    register_trace(G)
    _update_history(G)


def cmd_run(args: argparse.Namespace) -> int:
    G = build_graph(n=args.nodes, topology=args.topology, seed=args.seed)
    _attach_callbacks(G)
    validate_canon(G)
    gcanon = dict(DEFAULTS["GRAMMAR_CANON"])
    gcanon.update(_args_to_dict(args, prefix="grammar."))
    if hasattr(args, "grammar_canon") and args.grammar_canon is not None:
        gcanon["enabled"] = bool(args.grammar_canon)
    G.graph.setdefault("GRAMMAR_CANON", {}).update(gcanon)
    if args.glyph_hysteresis_window is not None:
        G.graph["GLYPH_HYSTERESIS_WINDOW"] = int(args.glyph_hysteresis_window)
    G.graph["glyph_selector"] = default_glyph_selector if args.selector == "basic" else parametric_glyph_selector
    G.graph["GAMMA"] = {
        "type": args.gamma_type,
        "beta": args.gamma_beta,
        "R0": args.gamma_R0,
    }

    if args.preset:
        program = get_preset(args.preset)
        play(G, program)
    else:
        steps = int(args.steps or 100)
        for _ in range(steps):
            step(G)

    if args.save_history:
        _save_json(args.save_history, G.graph.get("history", {}))
    if args.export_history_base:
        export_history(G, args.export_history_base, fmt=args.export_format)

    if args.summary:
        tg = Tg_global(G, normalize=True)
        lat = latency_series(G)
        print("Top glifos por Tg:", glyph_top(G, k=5))
        if lat["value"]:
            print("Latencia media:", sum(lat["value"]) / max(1, len(lat["value"])) )
    return 0


def cmd_sequence(args: argparse.Namespace) -> int:
    G = build_graph(n=args.nodes, topology=args.topology, seed=args.seed)
    _attach_callbacks(G)
    validate_canon(G)
    gcanon = dict(DEFAULTS["GRAMMAR_CANON"])
    gcanon.update(_args_to_dict(args, prefix="grammar."))
    if hasattr(args, "grammar_canon") and args.grammar_canon is not None:
        gcanon["enabled"] = bool(args.grammar_canon)
    G.graph.setdefault("GRAMMAR_CANON", {}).update(gcanon)
    if args.glyph_hysteresis_window is not None:
        G.graph["GLYPH_HYSTERESIS_WINDOW"] = int(args.glyph_hysteresis_window)
    G.graph["glyph_selector"] = default_glyph_selector if args.selector == "basic" else parametric_glyph_selector
    G.graph["GAMMA"] = {
        "type": args.gamma_type,
        "beta": args.gamma_beta,
        "R0": args.gamma_R0,
    }

    if args.preset:
        program = get_preset(args.preset)
    elif args.sequence_file:
        program = _load_sequence(args.sequence_file)
    else:
        program = seq("A’L", "E’N", "I’L", block("O’Z", "Z’HIR", "I’L", repeat=1), "R’A", "SH’A")

    play(G, program)

    if args.save_history:
        _save_json(args.save_history, G.graph.get("history", {}))
    if args.export_history_base:
        export_history(G, args.export_history_base, fmt=args.export_format)
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    G = build_graph(n=args.nodes, topology=args.topology, seed=args.seed)
    _attach_callbacks(G)
    validate_canon(G)
    G.graph.setdefault("GRAMMAR_CANON", DEFAULTS["GRAMMAR_CANON"]).update({"enabled": bool(args.grammar_canon)})
    G.graph["glyph_selector"] = default_glyph_selector if args.selector == "basic" else parametric_glyph_selector
    G.graph["GAMMA"] = {
        "type": args.gamma_type,
        "beta": args.gamma_beta,
        "R0": args.gamma_R0,
    }
    for _ in range(int(args.steps or 200)):
        step(G)

    tg = Tg_global(G, normalize=True)
    lat = latency_series(G)
    rose = sigma_rose(G)
    glifo = glifogram_series(G)

    out = {
        "Tg_global": tg,
        "latency_mean": (sum(lat["value"]) / max(1, len(lat["value"])) ) if lat["value"] else 0.0,
        "rose": rose,
        "glifogram": {k: v[:10] for k, v in glifo.items()},
    }
    if args.save:
        _save_json(args.save, out)
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="tnfr")
    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Correr escenario libre o preset y opcionalmente exportar history")
    p_run.add_argument("--nodes", type=int, default=24)
    p_run.add_argument("--topology", choices=["ring", "complete", "erdos"], default="ring")
    p_run.add_argument("--steps", type=int, default=200)
    p_run.add_argument("--seed", type=int, default=1)
    p_run.add_argument("--preset", type=str, default=None)
    p_run.add_argument("--save-history", dest="save_history", type=str, default=None)
    p_run.add_argument("--export-history-base", dest="export_history_base", type=str, default=None)
    p_run.add_argument("--export-format", dest="export_format", choices=["csv", "json"], default="json")
    p_run.add_argument("--summary", action="store_true")
    p_run.add_argument("--no-canon", dest="grammar_canon", action="store_false", default=True, help="Desactiva gramática canónica")
    p_run.add_argument("--grammar.enabled", dest="grammar_enabled", type=_str2bool, default=None)
    p_run.add_argument("--grammar.zhir_requires_oz_window", dest="grammar_zhir_requires_oz_window", type=int, default=None)
    p_run.add_argument("--grammar.zhir_dnfr_min", dest="grammar_zhir_dnfr_min", type=float, default=None)
    p_run.add_argument("--grammar.thol_min_len", dest="grammar_thol_min_len", type=int, default=None)
    p_run.add_argument("--grammar.thol_max_len", dest="grammar_thol_max_len", type=int, default=None)
    p_run.add_argument("--grammar.thol_close_dnfr", dest="grammar_thol_close_dnfr", type=float, default=None)
    p_run.add_argument("--grammar.si_high", dest="grammar_si_high", type=float, default=None)
    p_run.add_argument("--glyph.hysteresis_window", dest="glyph_hysteresis_window", type=int, default=None)
    p_run.add_argument("--selector", choices=["basic", "param"], default="basic")
    p_run.add_argument("--gamma-type", choices=list(GAMMA_REGISTRY.keys()), default="none")
    p_run.add_argument("--gamma-beta", type=float, default=0.0)
    p_run.add_argument("--gamma-R0", type=float, default=0.0)
    p_run.set_defaults(func=cmd_run)

    p_seq = sub.add_parser("sequence", help="Ejecutar una secuencia (preset o YAML/JSON)")
    p_seq.add_argument("--nodes", type=int, default=24)
    p_seq.add_argument("--topology", choices=["ring", "complete", "erdos"], default="ring")
    p_seq.add_argument("--seed", type=int, default=1)
    p_seq.add_argument("--preset", type=str, default=None)
    p_seq.add_argument("--sequence-file", type=str, default=None)
    p_seq.add_argument("--save-history", dest="save_history", type=str, default=None)
    p_seq.add_argument("--export-history-base", dest="export_history_base", type=str, default=None)
    p_seq.add_argument("--export-format", dest="export_format", choices=["csv", "json"], default="json")
    p_seq.add_argument("--gamma-type", choices=list(GAMMA_REGISTRY.keys()), default="none")
    p_seq.add_argument("--gamma-beta", type=float, default=0.0)
    p_seq.add_argument("--gamma-R0", type=float, default=0.0)
    p_seq.add_argument("--grammar.enabled", dest="grammar_enabled", type=_str2bool, default=None)
    p_seq.add_argument("--grammar.zhir_requires_oz_window", dest="grammar_zhir_requires_oz_window", type=int, default=None)
    p_seq.add_argument("--grammar.zhir_dnfr_min", dest="grammar_zhir_dnfr_min", type=float, default=None)
    p_seq.add_argument("--grammar.thol_min_len", dest="grammar_thol_min_len", type=int, default=None)
    p_seq.add_argument("--grammar.thol_max_len", dest="grammar_thol_max_len", type=int, default=None)
    p_seq.add_argument("--grammar.thol_close_dnfr", dest="grammar_thol_close_dnfr", type=float, default=None)
    p_seq.add_argument("--grammar.si_high", dest="grammar_si_high", type=float, default=None)
    p_seq.add_argument("--glyph.hysteresis_window", dest="glyph_hysteresis_window", type=int, default=None)
    p_seq.set_defaults(func=cmd_sequence)

    p_met = sub.add_parser("metrics", help="Correr breve y volcar métricas clave")
    p_met.add_argument("--nodes", type=int, default=24)
    p_met.add_argument("--topology", choices=["ring", "complete", "erdos"], default="ring")
    p_met.add_argument("--steps", type=int, default=300)
    p_met.add_argument("--seed", type=int, default=1)
    p_met.add_argument("--no-canon", dest="grammar_canon", action="store_false", default=True, help="Desactiva gramática canónica")
    p_met.add_argument("--selector", choices=["basic", "param"], default="basic")
    p_met.add_argument("--gamma-type", choices=list(GAMMA_REGISTRY.keys()), default="none")
    p_met.add_argument("--gamma-beta", type=float, default=0.0)
    p_met.add_argument("--gamma-R0", type=float, default=0.0)
    p_met.add_argument("--save", type=str, default=None)
    p_met.set_defaults(func=cmd_metrics)

    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        p.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
