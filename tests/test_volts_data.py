"""Synthetic-only controls for the pinned Volts research ingestion boundary."""

import gzip
import hashlib
import importlib.util
import sys
import zlib
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "benchmarks/volts_data.py"
_SPEC = importlib.util.spec_from_file_location("volts_ingestion", _PATH)
BENCH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = BENCH
_SPEC.loader.exec_module(BENCH)


def synthetic_frame():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {
            "Voltage": [3.0 - i / 100 for i in range(50)],
            "Time": [i / 10 for i in range(50)],
        },
        index=pd.Index(list(range(1, 51)), dtype="int64"),
    )


def synthetic_rdata(frame=None, *, name="Volts", version=2):
    rdata = pytest.importorskip("rdata")
    if rdata.__version__ != "1.1.0":
        pytest.skip("the research reader pins rdata1.1.0")
    frame = synthetic_frame() if frame is None else frame
    parsed = rdata.conversion.convert_python_to_r_data(
        {name: frame},
        format_version=version,
        file_type="rda",
    )
    return rdata.unparser.unparse_data(parsed, file_format="xdr", file_type="rda")


@pytest.mark.parametrize("version", [2, 3])
def test_synthetic_reader_preserves_full_rows_time_identity_and_metadata(version):
    expanded = synthetic_rdata(version=version)
    run, metadata = BENCH._decode_expanded(expanded)
    assert run.run_id == run.acquisition_id == "Stat2Data-Volts-single-trace"
    assert run.channel_ids == ("Voltage",)
    assert (run.value_unit, run.time_unit) == ("V", "s")
    assert run.timestamps == tuple(i / 10 for i in range(50))
    assert run.samples == (tuple(3.0 - i / 100 for i in range(50)),)
    assert metadata["expanded_bytes"] == len(expanded)
    assert metadata["expanded_sha256"] == hashlib.sha256(expanded).hexdigest()
    assert metadata["rdata_version"] == "1.1.0"
    assert metadata["serialization_format"] == version


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda f: f.iloc[:-1], "50 rows"),
        (lambda f: f[["Time", "Voltage"]], "columns"),
        (lambda f: f.rename(columns={"Time": "Seconds"}), "columns"),
        (lambda f: f.assign(Voltage=True), "real numeric"),
        (lambda f: f.assign(Voltage="1.0"), "real numeric"),
        (lambda f: f.assign(Voltage=complex(1, 1)), "real numeric"),
        (lambda f: f.assign(Voltage=float("nan")), "finite"),
        (lambda f: f.assign(Time=float("inf")), "finite"),
        (lambda f: f.assign(Time=1.0), "increasing"),
    ],
)
def test_converted_schema_rejects_without_dropping_or_coercing_rows(change, match):
    with pytest.raises(ValueError, match=match):
        BENCH._validated_run({"Volts": change(synthetic_frame())})


def test_multiple_or_wrong_named_datasets_are_not_silently_selected():
    frame = synthetic_frame()
    for payload in ({"Other": frame}, {"Volts": frame, "Other": frame}):
        with pytest.raises(ValueError, match="exactly one"):
            BENCH._validated_run(payload)


def test_rdata_wrong_dataset_name_is_rejected_after_synthetic_parse():
    with pytest.raises(ValueError, match="exactly one"):
        BENCH._decode_expanded(synthetic_rdata(name="Other"))


def test_unmarked_ascii_symbol_encoding_has_explicit_strict_default():
    rdata = pytest.importorskip("rdata")
    parsed = rdata.parser.parse_data(synthetic_rdata(), extension=".rda")
    parsed.object.tag.value.info.gp = 0
    expanded = rdata.unparser.unparse_data(parsed, file_type="rda")
    run, _ = BENCH._decode_expanded(expanded)
    assert run.run_id == BENCH.RUN_ID
    assert len(run.timestamps) == 50


@pytest.mark.parametrize("encoding_flags", [0, 4, 8, 64])
def test_non_ascii_metadata_is_rejected_without_replacement(encoding_flags):
    rdata = pytest.importorskip("rdata")
    parsed = rdata.parser.parse_data(synthetic_rdata(), extension=".rda")
    parsed.object.tag.value.info.gp = encoding_flags
    parsed.object.tag.value.value = b"V\xfflts"
    expanded = rdata.unparser.unparse_data(parsed, file_type="rda")
    with pytest.raises(ValueError, match="only ASCII"):
        BENCH._decode_expanded(expanded)


def test_custom_r_class_rejected_before_fallback_conversion():
    rdata = pytest.importorskip("rdata")
    parsed = rdata.conversion.convert_python_to_r_data(
        {"Volts": synthetic_frame()},
        format_version=2,
        file_type="rda",
    )
    frame = parsed.object.value[0]
    attrs = frame.attributes
    while attrs.info.type == rdata.parser.RObjectType.LIST:
        if attrs.tag.value.value == b"class":
            attrs.value[0].value[0].value = b"malicious.class"
            break
        attrs = attrs.value[1]
    else:
        pytest.fail("synthetic frame lacks its class")
    expanded = rdata.unparser.unparse_data(parsed, file_type="rda")
    with pytest.raises(Warning, match="Missing constructor"):
        BENCH._decode_expanded(expanded)


@pytest.mark.parametrize("compress", [gzip.compress, zlib.compress])
def test_bounded_compression_preserves_uncompressed_bytes(compress):
    expanded = synthetic_rdata()
    assert BENCH._bounded_expand(compress(expanded)) == expanded


@pytest.mark.parametrize(
    "make_raw,match",
    [
        (lambda: b"x" * 1025, "compressed"),
        (lambda: gzip.compress(b"x" * 65537), "expanded"),
        (lambda: gzip.compress(b"RDX2\nX\n")[:-1], "complete"),
        (lambda: gzip.compress(b"RDX2\nX\n") + b"junk", "trailing"),
        (lambda: gzip.compress(b"RDX2\nX\n") * 2, "trailing"),
        (lambda: gzip.compress(gzip.compress(b"RDX2\nX\n")), "nested"),
        (lambda: b"not compressed", "valid gzip/zlib"),
    ],
)
def test_compression_limits_are_enforced_before_library_parse(make_raw, match):
    with pytest.raises(ValueError, match=match):
        BENCH._bounded_expand(make_raw())


@pytest.mark.parametrize(
    "size,match", [(767, "SHA-256"), (766, "size"), (1025, "compressed")]
)
def test_identity_failure_precedes_any_decode(tmp_path, monkeypatch, size, match):
    path = tmp_path / "synthetic.rda"
    path.write_bytes(b"x" * size)
    monkeypatch.setattr(
        BENCH, "_bounded_expand", lambda _: pytest.fail("must not decompress")
    )
    monkeypatch.setattr(
        BENCH, "_rdata_module", lambda: pytest.fail("must not import reader")
    )
    with pytest.raises(ValueError, match=match):
        BENCH.load_volts(path)


def test_public_metadata_and_digest_for_identity_substituted_synthetic_fixture(
    tmp_path, monkeypatch
):
    expanded = synthetic_rdata()
    raw = gzip.compress(expanded)
    path = tmp_path / "synthetic.rda"
    path.write_bytes(raw)
    monkeypatch.setattr(BENCH, "SOURCE_BYTES", len(raw))
    monkeypatch.setattr(BENCH, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    run, metadata = BENCH.load_volts(path)
    assert len(run.timestamps) == 50
    assert metadata["source_bytes"] == len(raw)
    assert metadata["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert metadata["physical_status"] == "not_admitted"
    assert metadata["split_scope"] == "within_single_acquisition"


def test_missing_optional_dependency_has_actionable_error(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(BENCH.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="optional research dependency rdata==1.1.0"):
        BENCH._rdata_module()
