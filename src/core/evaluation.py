"""
Core orchestration for TNFR-Chess evaluation.

Pipeline:
  ChessTNFRGraphBuilder -> NodalDynamicsEngine -> TetradEvaluator -> aggregation per color.

Exposes a single DTO: PositionSnapshot
  - tetrad (Φ_s, |∇φ|, K_φ, ξ_C, C, Si, is_safe)
  - graph metrics per color (coherence/pressure/trend/mobility)
  - derived helpers (recommended_time, U2/U3/U4 flags)

Wrappers StructuralEvaluator and TelemetryView delegate to this orchestrator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import chess

from tetrad_evaluator import TetradEvaluator, TetradFields

ColorKey = str  # "white" | "black"


@dataclass
class GraphMetrics:
    coherence: float = 0.0  # how connected/focused this side is
    pressure: float = 0.0  # material pressure proxy
    trend: float = 0.0  # initiative proxy (center control / king pressure)
    mobility: float = 0.0  # normalized legal move count

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class PositionSnapshot:
    tetrad: TetradFields
    graph: Dict[ColorKey, GraphMetrics]
    recommended_time: float
    alert: bool
    u2_ok: bool  # convergence/boundedness (safety)
    u3_ok: bool  # resonant coupling (both sides have mobility)
    u4_ok: bool  # bifurcation guarded (pressure not spiking against low coherence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tetrad": asdict(self.tetrad),
            "graph": {k: v.to_dict() for k, v in self.graph.items()},
            "recommended_time": self.recommended_time,
            "alert": self.alert,
            "u2_ok": self.u2_ok,
            "u3_ok": self.u3_ok,
            "u4_ok": self.u4_ok,
        }


class ChessTNFRGraphBuilder:
    """Extract simple per-color graph metrics."""

    def build(self, board: chess.Board) -> Dict[ColorKey, GraphMetrics]:
        metrics: Dict[ColorKey, GraphMetrics] = {
            "white": GraphMetrics(),
            "black": GraphMetrics(),
        }

        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0,
        }

        counts = {"white": 0, "black": 0}
        pressure = {"white": 0.0, "black": 0.0}
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        center_control = {"white": 0, "black": 0}
        king_pressure = {"white": 0, "black": 0}

        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if not piece:
                continue
            color_key = "white" if piece.color == chess.WHITE else "black"
            counts[color_key] += 1
            pressure[color_key] += piece_values.get(piece.piece_type, 0.0)

            attacks = board.attacks(sq)
            if attacks and attacks & chess.SquareSet(center_squares):
                center_control[color_key] += 1

            opp_king_sq = board.king(not piece.color)
            if opp_king_sq is not None and opp_king_sq in attacks:
                king_pressure[color_key] += 1

        legal_moves = list(board.legal_moves)
        mobility_white = sum(
            1
            for m in legal_moves
            if board.piece_at(m.from_square)
            and board.piece_at(m.from_square).color == chess.WHITE
        )
        mobility_black = len(legal_moves) - mobility_white

        for color_key in ("white", "black"):
            gm = metrics[color_key]
            gm.pressure = pressure[color_key]
            gm.mobility = (
                mobility_white if color_key == "white" else mobility_black
            ) / max(1, len(legal_moves))
            gm.trend = (center_control[color_key] + king_pressure[color_key]) / max(
                1, counts[color_key]
            )

        total = max(1, counts["white"] + counts["black"])
        metrics["white"].coherence = (
            1.0 - abs(counts["white"] - counts["black"]) / total
        )
        metrics["black"].coherence = metrics["white"].coherence

        return metrics


class NodalDynamicsEngine:
    """Orchestrates TNFR evaluation for a board position."""

    def __init__(self) -> None:
        self.tetrad_eval = TetradEvaluator()
        self.graph_builder = ChessTNFRGraphBuilder()

    def compute_snapshot(self, board: chess.Board) -> PositionSnapshot:
        tetrad = self.tetrad_eval.evaluate(board)
        graph = self.graph_builder.build(board)

        recommended_time = max(0.5, 3.0 * (1.0 - tetrad.coherence))

        alert = not tetrad.is_safe

        u2_ok = tetrad.is_safe
        u3_ok = all(g.mobility > 0.05 for g in graph.values())
        u4_ok = not (
            tetrad.coherence < 0.2 and max(g.pressure for g in graph.values()) > 20.0
        )

        return PositionSnapshot(
            tetrad=tetrad,
            graph=graph,
            recommended_time=recommended_time,
            alert=alert,
            u2_ok=u2_ok,
            u3_ok=u3_ok,
            u4_ok=u4_ok,
        )

    def snapshot_from(self, board: chess.Board, *_: Any) -> PositionSnapshot:
        """Compatibility shim matching engine expectations."""

        return self.compute_snapshot(board)


class StructuralEvaluator:
    """Wrapper delegating to NodalDynamicsEngine (single entry point)."""

    def __init__(self) -> None:
        self.engine = NodalDynamicsEngine()

    def evaluate(self, board: chess.Board) -> PositionSnapshot:
        return self.engine.compute_snapshot(board)


class TelemetryView:
    """Light wrapper to expose snapshots to telemetry/loggers."""

    def __init__(self) -> None:
        self.engine = NodalDynamicsEngine()

    def snapshot(self, board: chess.Board) -> Dict[str, Any]:
        return self.engine.compute_snapshot(board).to_dict()


__all__ = [
    "GraphMetrics",
    "PositionSnapshot",
    "ChessTNFRGraphBuilder",
    "NodalDynamicsEngine",
    "StructuralEvaluator",
    "TelemetryView",
]
