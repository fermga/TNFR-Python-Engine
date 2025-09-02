"""Funciones auxiliares."""
from __future__ import annotations
from typing import Iterable, Sequence, Dict, Any, Callable, TypeVar, Mapping
from collections.abc import Collection
import logging
import math
from collections import deque, Counter
from itertools import islice
import heapq
from statistics import fmean, StatisticsError
import json
from json import JSONDecodeError
from pathlib import Path
from enum import Enum

try:  # pragma: no cover - dependencia opcional
    import yaml  # type: ignore
    from yaml import YAMLError  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None
    class YAMLError(Exception):  # type: ignore
        pass

from .constants import (
    DEFAULTS,
    ALIAS_VF,
    ALIAS_THETA,
    ALIAS_DNFR,
    ALIAS_dEPI,
    ALIAS_SI,
    ALIAS_EPI_KIND,
    ALIAS_D2EPI,
    get_param,
)

T = TypeVar("T")

__all__ = [
    "read_structured_file",
    "ensure_parent",
    "ensure_collection",
    "clamp",
    "clamp01",
    "list_mean",
    "angle_diff",
    "normalize_weights",
    "alias_get",
    "alias_set",
    "get_attr",
    "set_attr",
    "get_attr_str",
    "set_attr_str",
    "set_attr_with_max",
    "set_vf",
    "set_dnfr",
    "media_vecinal",
    "fase_media",
    "push_glifo",
    "reciente_glifo",
    "ensure_history",
    "last_glifo",
    "count_glyphs",
    "normalize_counter",
    "mix_groups",
    "register_callback",
    "invoke_callbacks",
    "compute_dnfr_accel_max",
    "compute_coherence",
    "compute_Si",
    "increment_edge_version",
    "CallbackEvent",
]


class CallbackEvent(str, Enum):
    """Eventos soportados para callbacks."""

    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_REMESH = "on_remesh"


_CALLBACK_EVENTS = tuple(e.value for e in CallbackEvent)

# -------------------------
# Entrada/salida estructurada
# -------------------------

def _parse_json(text: str) -> Any:
    """Parsea ``text`` como JSON."""
    return json.loads(text)


def _parse_yaml(text: str) -> Any:
    """Parsea ``text`` como YAML."""
    if not yaml:  # pragma: no cover - dependencia opcional
        raise RuntimeError("pyyaml no está instalado")
    return yaml.safe_load(text)


PARSERS: Dict[str, Callable[[str], Any]] = {
    ".json": _parse_json,
    ".yaml": _parse_yaml,
    ".yml": _parse_yaml,
}


def read_structured_file(path: Path) -> Any:
    """Lee un archivo JSON o YAML y devuelve los datos parseados."""
    suffix = path.suffix.lower()
    if suffix not in PARSERS:
        raise ValueError(f"Extensión de archivo no soportada: {suffix}")
    parser = PARSERS[suffix]
    if not path.is_file():
        raise ValueError(f"El archivo no existe: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except PermissionError as e:
        raise ValueError(f"Permiso denegado al leer {path}: {e}") from e
    except FileNotFoundError as e:  # pragma: no cover - carrera improbable
        raise ValueError(f"El archivo no existe: {path}") from e

    try:
        return parser(text)
    except JSONDecodeError as e:
        raise ValueError(f"Error al parsear archivo JSON en {path}: {e}") from e
    except YAMLError as e:
        raise ValueError(f"Error al parsear archivo YAML en {path}: {e}") from e
    except RuntimeError as e:
        raise ValueError(
            f"Dependencia faltante al parsear {path}: {e}"
        ) from e


def ensure_parent(path: str | Path) -> None:
    """Crea el directorio padre de ``path`` si hace falta."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# -------------------------
# Iterables y colecciones
# -------------------------

def ensure_collection(it: Iterable[T], *, max_materialize: int | None = None) -> Collection[T]:
    """Devuelve ``it`` si ya es ``Collection`` o materializa en ``tuple`` en caso contrario.

    Cadenas de texto y objetos *bytes* se tratan como un único elemento en
    lugar de iterarse carácter a carácter. En esos casos se devuelve una
    ``tuple`` de un solo elemento. Si ``it`` no es iterable se lanza
    ``TypeError``.

    Parameters
    ----------
    max_materialize:
        Número máximo de elementos a materializar cuando ``it`` no es una
        colección. Si el iterable produce más de ``max_materialize`` elementos
        se lanza :class:`ValueError`. ``None`` (por defecto) implica sin
        límite.

    Notes
    -----
    Materializar un iterable potencialmente grande puede suponer un coste de
    memoria elevado. El parámetro ``max_materialize`` permite acotar este
    coste.
    """

    if isinstance(it, Collection) and not isinstance(it, (str, bytes, bytearray)):
        return it
    if isinstance(it, (str, bytes, bytearray)):
        return (it,)
    try:
        if max_materialize is None:
            return tuple(it)
        data = tuple(islice(it, max_materialize + 1))
        if len(data) > max_materialize:
            raise ValueError(
                f"Iterable materialization exceeded {max_materialize} items"
            )
        return data
    except TypeError as exc:  # pragma: no cover - Defensive; unlikely with type hints
        raise TypeError(f"{it!r} is not iterable") from exc

# -------------------------
# Utilidades numéricas
# -------------------------

def clamp(x: float, a: float, b: float) -> float:
    """Constriñe ``x`` al intervalo cerrado [a, b]."""
    return a if x < a else b if x > b else x


def clamp01(x: float) -> float:
    """Ataja ``x`` a la banda [0, 1]."""
    return clamp(x, 0.0, 1.0)


def list_mean(xs: Iterable[float], default: float = 0.0) -> float:
    """Promedio aritmético o ``default`` si ``xs`` está vacío."""
    try:
        return fmean(xs)
    except StatisticsError:
        return default


def _wrap_angle(a: float) -> float:
    """Envuelve ángulo a (-π, π]."""
    pi = math.pi
    a = (a + pi) % (2 * pi) - pi
    return a


def angle_diff(a: float, b: float) -> float:
    """Diferencia mínima entre ``a`` y ``b`` en (-π, π]."""
    return _wrap_angle(a - b)


def normalize_weights(
    dict_like: Dict[str, Any],
    keys: Iterable[str],
    default: float = 0.0,
    *,
    error_on_negative: bool = False,
) -> Dict[str, float]:
    """Devuelve ``dict`` de ``keys`` normalizadas a sumatorio 1.

    La función se divide en tres pasos: conversión de valores, validación y
    normalización final. Cada clave de ``keys`` se obtiene de ``dict_like`` y se
    convierte a ``float``. Si la suma de los valores es <= 0 se asignan
    proporciones uniformes entre todas las claves.

    Parameters
    ----------
    error_on_negative:
        Si es ``True`` se lanza :class:`ValueError` ante valores negativos o
        pesos no numéricos. En caso contrario se registra una advertencia y se
        utiliza el valor ``default``.
    """
    keys = list(keys)
    default_float = float(default)

    weights = _convert_weights(dict_like, keys, default_float, error_on_negative)
    _validate_weights(weights, error_on_negative)
    return _normalize_distribution(weights, keys)


def _convert_weights(
    dict_like: Mapping[str, Any],
    keys: Iterable[str],
    default_float: float,
    error_on_negative: bool,
) -> Dict[str, float]:
    """Convierte ``dict_like`` a ``float`` usando ``_convert_value``."""

    def _to_float(k: str) -> float:
        val = dict_like.get(k, default_float)
        ok, converted = _convert_value(
            val,
            float,
            strict=error_on_negative,
            key=k,
            log_level=logging.WARNING,
        )
        return converted if ok and converted is not None else default_float

    return {k: _to_float(k) for k in keys}


def _validate_weights(weights: Mapping[str, float], error_on_negative: bool) -> None:
    """Valida que no existan pesos negativos."""

    negatives = {k: v for k, v in weights.items() if v < 0}
    if not negatives:
        return
    if error_on_negative:
        raise ValueError(f"Pesos negativos detectados: {negatives}")
    logging.warning("Pesos negativos detectados: %s", negatives)


def _normalize_distribution(
    weights: Mapping[str, float], keys: Sequence[str]
) -> Dict[str, float]:
    """Normaliza ``weights`` para que su sumatorio sea 1."""

    total = math.fsum(weights.values())
    n = len(keys)
    if total <= 0:
        if n == 0:
            return {}
        uniform = 1.0 / n
        return {k: uniform for k in keys}
    return {k: v / total for k, v in weights.items()}


# -------------------------
# Acceso a atributos con alias
# -------------------------


def _convert_value(
    value: Any,
    conv: Callable[[Any], T],
    *,
    strict: bool = False,
    key: str | None = None,
    log_level: int | None = None,
) -> tuple[bool, T | None]:
    """Intenta convertir ``value`` usando ``conv`` manejando errores.

    ``log_level`` controla el nivel de logging cuando la conversión falla en
    modo laxo. Por defecto se usa ``logging.ERROR`` si ``strict`` es ``True`` y
    ``logging.DEBUG`` en caso contrario.
    """
    try:
        return True, conv(value)
    except (ValueError, TypeError) as exc:
        level = log_level if log_level is not None else (
            logging.ERROR if strict else logging.DEBUG
        )
        if key is not None:
            logging.log(level, "No se pudo convertir el valor para %r: %s", key, exc)
        else:
            logging.log(level, "No se pudo convertir el valor: %s", exc)
        if strict:
            raise
        return False, None


def alias_get(
    d: Dict[str, Any],
    aliases: Sequence[str],
    conv: Callable[[Any], T],
    *,
    default: Any | None = None,
    strict: bool = False,
    log_level: int | None = None,
) -> T | None:
    """Busca en ``d`` la primera clave de ``aliases`` y retorna el valor convertido.

    ``aliases`` debe ser una secuencia (idealmente una tupla) de claves. No se
    realiza ninguna conversión interna, por lo que pasar una cadena única
    resultará en un error.

    Si ninguna de las claves está presente o la conversión falla, devuelve
    ``default`` convertido (o ``None`` si ``default`` es ``None``).

    ``log_level`` permite ajustar el nivel de logging cuando la conversión
    falla en modo laxo.
    """
    if isinstance(aliases, str):
        raise TypeError("'aliases' must be a sequence of strings")
    if not aliases:
        raise ValueError("'aliases' must contain at least one key")
    for key in aliases:
        if key in d:
            ok, val = _convert_value(
                d[key], conv, strict=strict, key=key, log_level=log_level
            )
            if ok:
                return val
    if default is None:
        return None
    ok, val = _convert_value(
        default, conv, strict=strict, key="default", log_level=log_level
    )
    return val if ok else None


def alias_set(
    d: Dict[str, Any],
    aliases: Sequence[str],
    conv: Callable[[Any], T],
    value: Any,
) -> T:
    """Asigna ``value`` convertido a la primera clave disponible de ``aliases``.

    ``aliases`` debe ser una secuencia (idealmente una tupla) de claves y no se
    transforma internamente.
    """
    if isinstance(aliases, str):
        raise TypeError("'aliases' must be a sequence of strings")
    if not aliases:
        raise ValueError("'aliases' must contain at least one key")
    _, val = _convert_value(value, conv, strict=True)
    for key in aliases:
        if key in d:
            d[key] = val
            return val
    key = aliases[0]
    d[key] = val
    return val


def get_attr(
    d: Dict[str, Any],
    aliases: Sequence[str],
    default: float = 0.0,
    *,
    strict: bool = False,
    log_level: int | None = None,
) -> float:
    """Obtiene un atributo numérico usando :func:`alias_get`.

    ``aliases`` debe ser una secuencia de claves (idealmente una tupla).
    """
    return alias_get(
        d, aliases, float, default=default, strict=strict, log_level=log_level
    )


def set_attr(d, aliases: Sequence[str], value: float) -> float:
    """Establece un atributo numérico usando :func:`alias_set`.

    ``aliases`` debe ser una secuencia de claves (idealmente una tupla).
    """
    return alias_set(d, aliases, float, value)


def get_attr_str(
    d: Dict[str, Any],
    aliases: Sequence[str],
    default: str = "",
    *,
    strict: bool = False,
    log_level: int | None = None,
) -> str:
    """Obtiene un atributo de texto usando :func:`alias_get`.

    ``aliases`` debe ser una secuencia de claves (idealmente una tupla).
    """
    return alias_get(
        d, aliases, str, default=default, strict=strict, log_level=log_level
    )


def set_attr_str(d, aliases: Sequence[str], value: str) -> str:
    """Establece un atributo de texto usando :func:`alias_set`.

    ``aliases`` debe ser una secuencia de claves (idealmente una tupla).
    """
    return alias_set(d, aliases, str, value)

# Retrocompatibilidad con nombres anteriores
_get_attr = get_attr
_set_attr = set_attr
_get_attr_str = get_attr_str
_set_attr_str = set_attr_str


# -------------------------
# Máximos globales con caché
# -------------------------

def _recompute_abs_max(G, aliases: Sequence[str]):
    """Recalcula y retorna ``(max_val, node)`` para ``aliases``."""
    node = max(
        G.nodes(),
        key=lambda m: abs(get_attr(G.nodes[m], aliases, 0.0)),
        default=None,
    )
    max_val = abs(get_attr(G.nodes[node], aliases, 0.0)) if node is not None else 0.0
    return max_val, node


def _update_cached_abs_max(
    G, aliases: Sequence[str], n, value, *, key: str
) -> None:
    """Actualiza ``G.graph[key]`` y ``G.graph[f"{key}_node"]``."""
    node_key = f"{key}_node"
    val = abs(value)
    cur = float(G.graph.get(key, 0.0))
    cur_node = G.graph.get(node_key)
    if val >= cur:
        G.graph[key] = val
        G.graph[node_key] = n
    elif cur_node == n and val < cur:
        max_val, max_node = _recompute_abs_max(G, aliases)
        G.graph[key] = max_val
        G.graph[node_key] = max_node


def set_attr_with_max(
    G, n, aliases: Sequence[str], value: float, *, cache: str
) -> None:
    """Asigna ``value`` al atributo indicado y actualiza el máximo global."""
    val = float(value)
    set_attr(G.nodes[n], aliases, val)
    _update_cached_abs_max(G, aliases, n, val, key=cache)


def set_vf(G, n, value: float) -> None:
    """Asigna ``νf`` y actualiza el máximo global."""
    set_attr_with_max(G, n, ALIAS_VF, value, cache="_vfmax")


def set_dnfr(G, n, value: float) -> None:
    """Asigna ``ΔNFR`` y actualiza el máximo global."""
    set_attr_with_max(G, n, ALIAS_DNFR, value, cache="_dnfrmax")


# -------------------------
# Normalizadores de ΔNFR y aceleración
# -------------------------

def compute_dnfr_accel_max(G) -> dict:
    """Calcula los máximos absolutos de |ΔNFR| y |d²EPI/dt²|.

    Devuelve un diccionario con las claves ``dnfr_max`` y ``accel_max``.
    Si el grafo no tiene nodos, ambos valores serán ``0.0``.
    """

    dnfr_max = 0.0
    accel_max = 0.0
    for _, nd in G.nodes(data=True):
        dnfr_max = max(dnfr_max, abs(get_attr(nd, ALIAS_DNFR, 0.0)))
        accel_max = max(accel_max, abs(get_attr(nd, ALIAS_D2EPI, 0.0)))
    return {"dnfr_max": float(dnfr_max), "accel_max": float(accel_max)}

# -------------------------
# Coherencia global
# -------------------------

def compute_coherence(G) -> float:
    """Calcula la coherencia global C(t) a partir de ΔNFR y dEPI."""
    dnfr_sum = 0.0
    depi_sum = 0.0
    count = 0
    for n in G.nodes():
        nd = G.nodes[n]
        dnfr_sum += abs(get_attr(nd, ALIAS_DNFR, 0.0))
        depi_sum += abs(get_attr(nd, ALIAS_dEPI, 0.0))
        count += 1
    if count:
        dnfr_mean = dnfr_sum / count
        depi_mean = depi_sum / count
    else:
        dnfr_mean = depi_mean = 0.0
    return 1.0 / (1.0 + dnfr_mean + depi_mean)

# -------------------------
# Estadísticos vecinales
# -------------------------

def media_vecinal(G, n, aliases: Sequence[str], default: float = 0.0) -> float:
    """Media del atributo indicado por ``aliases`` en los vecinos de ``n``."""
    vals = (get_attr(G.nodes[v], aliases, default) for v in G.neighbors(n))
    return list_mean(vals, default)


def fase_media(obj, n=None) -> float:
    """Promedio circular de las fases vecinales.

    Acepta un :class:`NodoProtocol` o un par ``(G, n)`` de ``networkx``. En el
    segundo caso se envuelve en :class:`NodoNX` para reutilizar la misma lógica.
    """

    if n is not None:
        from .node import NodoNX  # importación local para evitar ciclo
        node = NodoNX(obj, n)
    else:
        node = obj  # se asume NodoProtocol

    x = y = 0.0
    count = 0
    for v in node.neighbors():
        th = getattr(v, "theta", 0.0)
        x += math.cos(th)
        y += math.sin(th)
        count += 1
    if count == 0:
        return getattr(node, "theta", 0.0)
    return math.atan2(y, x)


# -------------------------
# Historial de glifos por nodo
# -------------------------

def push_glifo(nd: Dict[str, Any], glifo: str, window: int) -> None:
    """Añade ``glifo`` al historial del nodo con tamaño máximo ``window``."""
    hist = nd.get("hist_glifos")
    if hist is None or hist.maxlen != window:
        hist = deque(hist or [], maxlen=window)
        nd["hist_glifos"] = hist
    hist.append(str(glifo))


def reciente_glifo(nd: Dict[str, Any], glifo: str, ventana: int) -> bool:
    """Indica si ``glifo`` apareció en las últimas ``ventana`` emisiones"""
    hist = nd.get("hist_glifos")
    gl = str(glifo)
    if ventana < 0:
        raise ValueError("ventana debe ser >= 0")
    if hist and ventana > 0:
        for reciente in islice(reversed(hist), ventana):
            if gl == reciente:
                return True
    # fallback al glifo dominante actual
    return get_attr_str(nd, ALIAS_EPI_KIND, "") == gl

# -------------------------
# Utilidades de historial global
# -------------------------

class HistoryDict(dict):
    """Dict especializado que crea deques acotados para series y cuenta usos."""

    def __init__(self, data: Dict[str, Any] | None = None, *, maxlen: int = 0):
        super().__init__(data or {})
        self._maxlen = maxlen
        self._counts: Dict[str, int] = {}
        self._heap: list[tuple[int, str]] = []
        if self._maxlen > 0:
            for k, v in list(self.items()):
                if isinstance(v, list):
                    self[k] = deque(v, maxlen=self._maxlen)
                self._counts.setdefault(k, 0)
                heapq.heappush(self._heap, (0, k))

    def _compact_heap(self) -> None:
        """Elimina entradas obsoletas de ``_heap``.

        Las entradas quedan obsoletas cuando la cuenta almacenada no coincide
        con ``_counts`` o cuando la clave ya no existe en el diccionario. Para
        evitar que la lista crezca indefinidamente se reconstruye filtrando las
        entradas válidas y luego se re-heapifica.
        """

        self._heap = [
            (cnt, k)
            for cnt, k in self._heap
            if k in self and self._counts.get(k) == cnt
        ]
        heapq.heapify(self._heap)

    def _maybe_compact(self) -> None:
        """Compacta ``_heap`` si supera un umbral razonable."""

        if len(self._heap) > len(self) * 2:
            self._compact_heap()

    def _increment(self, key: str) -> None:
        cnt = self._counts.get(key, 0) + 1
        self._counts[key] = cnt
        heapq.heappush(self._heap, (cnt, key))
        self._maybe_compact()

    def __getitem__(self, key):  # type: ignore[override]
        val = super().__getitem__(key)
        self._increment(key)
        return val

    def get(self, key, default=None):  # type: ignore[override]
        return super().get(key, default)

    def tracked_get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __setitem__(self, key, value):  # type: ignore[override]
        super().__setitem__(key, value)
        self._counts.setdefault(key, 0)
        heapq.heappush(self._heap, (self._counts[key], key))
        self._maybe_compact()

    def setdefault(self, key, default=None):  # type: ignore[override]
        if self._maxlen > 0 and isinstance(default, list):
            default = deque(default, maxlen=self._maxlen)
        if key in self:
            val = self[key]
        else:
            val = super().setdefault(key, default)
            if self._maxlen > 0 and isinstance(val, list):
                val = deque(val, maxlen=self._maxlen)
                super().__setitem__(key, val)
        self._increment(key)
        return val

    def pop_least_used(self) -> Any:
        while self._heap:
            cnt, key = heapq.heappop(self._heap)
            if self._counts.get(key) == cnt and key in self:
                self._counts.pop(key, None)
                return super().pop(key)
        raise KeyError("HistoryDict is empty")


def ensure_history(G) -> Dict[str, Any]:
    """Garantiza ``G.graph['history']`` y la devuelve.

    Si ``HISTORY_MAXLEN`` > 0, cada serie se almacena en un ``deque`` con ese
    límite y se eliminan claves poco usadas cuando el historial crece por encima
    del máximo permitido.
    """

    maxlen = int(G.graph.get("HISTORY_MAXLEN", get_param(G, "HISTORY_MAXLEN")))
    hist = G.graph.get("history")
    if not isinstance(hist, HistoryDict) or hist._maxlen != maxlen:
        hist = HistoryDict(hist, maxlen=maxlen)
        G.graph["history"] = hist
    if maxlen > 0:
        while len(hist) > maxlen:
            try:
                hist.pop_least_used()
            except KeyError:
                break
    return hist


def last_glifo(nd: Dict[str, Any]) -> str | None:
    """Retorna el glifo más reciente del nodo o ``None``."""
    kind = get_attr_str(nd, ALIAS_EPI_KIND, "")
    if kind:
        return kind
    hist = nd.get("hist_glifos")
    if not hist:
        return None
    try:
        return hist[-1]
    except IndexError:
        return None


def count_glyphs(
    G, window: int | None = None, *, last_only: bool = False
) -> Counter:
    """Cuenta glifos recientes en la red.

    Si ``last_only`` es ``True`` cuenta solo el último glifo de cada nodo. En
    caso contrario se usa el historial ``hist_glifos`` limitado a los últimos
    ``window`` elementos por nodo (o a todo el deque si ``window`` es ``None``).
    """
    counts: Counter[str] = Counter()
    for _, nd in G.nodes(data=True):
        if last_only:
            g = last_glifo(nd)
            seq = [g] if g else []
        else:
            hist = nd.get("hist_glifos")
            if not hist:
                continue
            if window is not None and window > 0:
                window_int = int(window)
                seq = islice(reversed(hist), window_int)
            else:
                seq = hist
        counts.update(seq)
    return counts


def normalize_counter(counts: Mapping[str, int]) -> tuple[Dict[str, float], int]:
    """Normaliza un ``Counter`` y devuelve proporciones y total.

    Si la suma total es cero, retorna ``({}, 0)``.
    """
    total = sum(counts.values())
    if total <= 0:
        return {}, 0
    dist = {k: v / total for k, v in counts.items() if v}
    return dist, total


def mix_groups(
    dist: Mapping[str, float],
    groups: Mapping[str, Iterable[str]],
    *,
    prefix: str = "_",
) -> Dict[str, float]:
    """Agrega valores de ``dist`` según agrupaciones.

    ``groups`` debe mapear nombres de grupo a iterables de claves presentes en
    ``dist``. Cada grupo se añadirá al resultado con el nombre
    ``prefix + label``.
    """
    out: Dict[str, float] = dict(dist)
    for label, keys in groups.items():
        out[f"{prefix}{label}"] = sum(dist.get(k, 0.0) for k in keys)
    return out

# -------------------------
# Callbacks Γ(R)
# -------------------------

def _ensure_callbacks(G):
    """Garantiza la estructura de callbacks en G.graph."""
    cbs = G.graph.setdefault(
        "callbacks", {k: [] for k in _CALLBACK_EVENTS}
    )
    # normaliza claves por si vienen incompletas
    for k in _CALLBACK_EVENTS:
        cbs.setdefault(k, [])
    return cbs

def register_callback(
    G,
    event: CallbackEvent | str,
    func=None,
    *,
    name: str | None = None,
):
    """Registra ``func`` como callback del ``event`` indicado.

    ``func`` puede pasarse como función o como tupla ``(name, func)``. En el
    primer caso se convertirá a dicha tupla antes de almacenarse. Si ya existe
    un callback con el mismo nombre o función para el evento, será reemplazado
    en lugar de añadirse una entrada duplicada.
    """
    if event not in _CALLBACK_EVENTS:
        raise ValueError(f"Evento desconocido: {event}")
    if func is None:
        raise TypeError("func es obligatorio")
    cbs = _ensure_callbacks(G)

    if isinstance(func, tuple):
        cb_name, func = func
    else:
        cb_name = name or getattr(func, "__name__", None)

    new_cb = (cb_name, func)

    # evita duplicados por nombre o función reemplazando la entrada existente
    for i, (existing_name, existing_fn) in enumerate(cbs[event]):
        if existing_fn is func or (cb_name is not None and existing_name == cb_name):
            cbs[event][i] = new_cb
            break
    else:
        cbs[event].append(new_cb)

    return func

def invoke_callbacks(G, event: CallbackEvent | str, ctx: dict | None = None):
    """Invoca todos los callbacks registrados para ``event`` con el contexto ``ctx``.

    Los callbacks se almacenan como tuplas ``(name, func)`` y se invocan en orden
    de registro.
    """
    cbs = _ensure_callbacks(G).get(event, [])
    strict = bool(G.graph.get("CALLBACKS_STRICT", DEFAULTS["CALLBACKS_STRICT"]))
    ctx = ctx or {}
    for name, fn in list(cbs):
        try:
            fn(G, ctx)
        except (KeyError, ValueError, TypeError) as e:
            if strict:
                raise
            G.graph.setdefault("_callback_errors", []).append({
                "event": event,
                "step": ctx.get("step"),
                "error": repr(e),
                "fn": repr(fn),
                "name": name,
            })

# -------------------------
# Índice de sentido (Si)
# -------------------------

def compute_Si(G, *, inplace: bool = True) -> Dict[Any, float]:
    """Calcula Si por nodo y lo escribe en G.nodes[n]["Si"].

    Fórmula:
        Si = α·νf_norm + β·(1 - disp_fase_local) + γ·(1 - |ΔNFR|/max|ΔNFR|)
    También guarda en ``G.graph`` los pesos normalizados y la
    sensibilidad parcial (∂Si/∂componente).
    """
    neighbors = {n: list(G.neighbors(n)) for n in G}
    w = {**DEFAULTS["SI_WEIGHTS"], **G.graph.get("SI_WEIGHTS", {})}
    weights = normalize_weights(w, ("alpha", "beta", "gamma"), default=0.0)
    alpha = weights["alpha"]
    beta = weights["beta"]
    gamma = weights["gamma"]
    G.graph["_Si_weights"] = weights
    G.graph["_Si_sensitivity"] = {
        "dSi_dvf_norm": alpha,
        "dSi_ddisp_fase": -beta,
        "dSi_ddnfr_norm": -gamma,
    }

    # Normalización de νf y ΔNFR en red usando máximos cacheados
    vfmax = G.graph.get("_vfmax")
    if vfmax is None:
        vfmax, vf_node = _recompute_abs_max(G, ALIAS_VF)
        G.graph.setdefault("_vfmax", vfmax)
        G.graph.setdefault("_vfmax_node", vf_node)
    dnfrmax = G.graph.get("_dnfrmax")
    if dnfrmax is None:
        dnfrmax, dnfr_node = _recompute_abs_max(G, ALIAS_DNFR)
        G.graph.setdefault("_dnfrmax", dnfrmax)
        G.graph.setdefault("_dnfrmax_node", dnfr_node)
    vfmax = 1.0 if vfmax == 0 else vfmax
    dnfrmax = 1.0 if dnfrmax == 0 else dnfrmax

    # precálculo de cosenos y senos de cada nodo
    cos_th: Dict[Any, float] = {}
    sin_th: Dict[Any, float] = {}
    thetas: Dict[Any, float] = {}
    for n, nd in G.nodes(data=True):
        th = get_attr(nd, ALIAS_THETA, 0.0)
        thetas[n] = th
        cos_th[n] = math.cos(th)
        sin_th[n] = math.sin(th)

    out: Dict[Any, float] = {}
    for n, nd in G.nodes(data=True):
        vf = get_attr(nd, ALIAS_VF, 0.0)
        vf_norm = clamp01(abs(vf) / vfmax)

        # dispersión de fase local utilizando vecinos precomputados
        th_i = thetas[n]
        neigh = neighbors[n]
        deg = len(neigh)
        if deg:
            sum_cos = sum(cos_th[v] for v in neigh)
            sum_sin = sum(sin_th[v] for v in neigh)
            mean_cos = sum_cos / deg
            mean_sin = sum_sin / deg
            th_bar = math.atan2(mean_sin, mean_cos)
        else:
            th_bar = th_i
        disp_fase = abs(angle_diff(th_i, th_bar)) / math.pi  # [0,1]

        dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
        dnfr_norm = clamp01(abs(dnfr) / dnfrmax)

        Si = alpha*vf_norm + beta*(1.0 - disp_fase) + gamma*(1.0 - dnfr_norm)
        Si = clamp01(Si)
        out[n] = Si
        if inplace:
            set_attr(nd, ALIAS_SI, Si)
    return out


def increment_edge_version(G: Any) -> None:
    """Incrementa el contador de versión de aristas en ``G.graph``.

    Acepta un ``nx.Graph`` o un diccionario que actúe como ``G.graph`` y
    actualiza ``"_edge_version"`` para invalidar caches dependientes de las
    aristas.
    """
    graph = G.graph if hasattr(G, "graph") else G
    graph["_edge_version"] = int(graph.get("_edge_version", 0)) + 1
