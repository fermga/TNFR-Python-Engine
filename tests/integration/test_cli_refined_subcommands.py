"""Tests for refined CLI subcommands (math.run, epi.validate) and presets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tnfr.cli import main


class TestMathRunSubcommand:
    """Tests for the ``tnfr math.run`` subcommand."""

    def test_math_run_help(self, capsys):
        """Test that math.run --help displays correctly."""
        with pytest.raises(SystemExit) as exc_info:
            main(["math.run", "--help"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "math.run" in captured.out
        assert "Mathematical dynamics" in captured.out
        assert "--math-dimension" in captured.out
        assert "Examples:" in captured.out

    def test_math_run_basic_execution(self, capsys):
        """Test basic math.run execution with minimal parameters."""
        rc = main([
            "math.run",
            "--nodes", "3",
            "--steps", "2",
            "--topology", "ring",
        ])
        assert rc == 0

        captured = capsys.readouterr()
        assert "[MATH.RUN]" in captured.out or "[MATH]" in captured.out

    def test_math_run_with_preset(self, capsys):
        """Test math.run with a preset."""
        rc = main([
            "math.run",
            "--preset", "resonant_bootstrap",
            "--nodes", "3",
            "--steps", "2",
        ])
        assert rc == 0

        captured = capsys.readouterr()
        # Should complete successfully
        assert rc == 0

    def test_math_run_with_dimension(self, capsys):
        """Test math.run with custom Hilbert dimension."""
        rc = main([
            "math.run",
            "--nodes", "3",
            "--steps", "2",
            "--math-dimension", "5",
        ])
        assert rc == 0

    def test_math_run_with_coherence_spectrum(self, capsys):
        """Test math.run with custom coherence spectrum."""
        rc = main([
            "math.run",
            "--nodes", "3",
            "--steps", "2",
            "--math-coherence-spectrum", "1.0", "0.8", "0.6",
        ])
        assert rc == 0

    def test_math_run_invalid_preset(self, capsys):
        """Test math.run with invalid preset shows error."""
        rc = main([
            "math.run",
            "--preset", "nonexistent_preset",
            "--nodes", "3",
        ])
        assert rc == 1

        captured = capsys.readouterr()
        assert "Unknown preset" in captured.out
        assert "Available presets" in captured.out


class TestEpiValidateSubcommand:
    """Tests for the ``tnfr epi.validate`` subcommand."""

    def test_epi_validate_help(self, capsys):
        """Test that epi.validate --help displays correctly."""
        with pytest.raises(SystemExit) as exc_info:
            main(["epi.validate", "--help"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "epi.validate" in captured.out
        assert "Validation options" in captured.out
        assert "--check-coherence" in captured.out
        assert "--check-frequency" in captured.out
        assert "--check-phase" in captured.out
        assert "Examples:" in captured.out

    def test_epi_validate_basic_execution(self, capsys):
        """Test basic epi.validate execution."""
        rc = main([
            "epi.validate",
            "--nodes", "3",
            "--steps", "2",
            "--topology", "ring",
        ])
        # Should pass validation
        assert rc == 0

        captured = capsys.readouterr()
        assert "[EPI.VALIDATE]" in captured.out
        assert "Validation Summary" in captured.out

    def test_epi_validate_with_preset(self, capsys):
        """Test epi.validate with a preset."""
        rc = main([
            "epi.validate",
            "--preset", "resonant_bootstrap",
            "--nodes", "3",
            "--steps", "2",
        ])
        assert rc == 0

        captured = capsys.readouterr()
        assert "[EPI.VALIDATE]" in captured.out

    def test_epi_validate_with_tolerance(self, capsys):
        """Test epi.validate with custom tolerance."""
        rc = main([
            "epi.validate",
            "--nodes", "3",
            "--steps", "2",
            "--tolerance", "1e-8",
        ])
        assert rc == 0

        captured = capsys.readouterr()
        assert "[EPI.VALIDATE]" in captured.out

    def test_epi_validate_checks_coherence(self, capsys):
        """Test that epi.validate reports coherence checks."""
        rc = main([
            "epi.validate",
            "--nodes", "5",
            "--steps", "3",
            "--check-coherence",
        ])
        assert rc == 0

        captured = capsys.readouterr()
        assert "Coherence" in captured.out or "[PASS]" in captured.out or "[SKIP]" in captured.out

    def test_epi_validate_checks_frequency(self, capsys):
        """Test that epi.validate reports frequency checks."""
        rc = main([
            "epi.validate",
            "--nodes", "5",
            "--steps", "3",
            "--check-frequency",
        ])
        assert rc == 0

        captured = capsys.readouterr()
        # Should mention frequency or show pass/skip
        output = captured.out
        assert "frequency" in output.lower() or "[PASS]" in output or "[SKIP]" in output

    def test_epi_validate_invalid_preset(self, capsys):
        """Test epi.validate with invalid preset shows error."""
        rc = main([
            "epi.validate",
            "--preset", "invalid_preset_name",
        ])
        assert rc == 1

        captured = capsys.readouterr()
        assert "Unknown preset" in captured.out


class TestEnhancedHelp:
    """Tests for enhanced CLI help messages."""

    def test_main_help_has_examples(self, capsys):
        """Test that main help includes common examples."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "Common examples:" in captured.out
        assert "math.run" in captured.out
        assert "epi.validate" in captured.out
        assert "presets/" in captured.out

    def test_main_help_lists_all_subcommands(self, capsys):
        """Test that main help lists all available subcommands."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "run" in captured.out
        assert "math.run" in captured.out
        assert "epi.validate" in captured.out
        assert "sequence" in captured.out
        assert "metrics" in captured.out

    def test_run_help_mentions_presets(self, capsys):
        """Test that run help mentions available presets."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--help"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "Available presets" in captured.out
        assert "resonant_bootstrap" in captured.out


class TestYAMLPresets:
    """Tests for YAML preset files."""

    def test_preset_files_exist(self):
        """Test that preset YAML files exist."""
        presets_dir = Path("presets")
        assert presets_dir.exists()
        assert presets_dir.is_dir()

        expected_presets = [
            "resonant_bootstrap.yaml",
            "contained_mutation.yaml",
            "coupling_exploration.yaml",
            "fractal_expand.yaml",
            "fractal_contract.yaml",
        ]

        for preset_name in expected_presets:
            preset_path = presets_dir / preset_name
            assert preset_path.exists(), f"Preset {preset_name} should exist"

    def test_preset_readme_exists(self):
        """Test that presets README exists."""
        readme_path = Path("presets") / "README.md"
        assert readme_path.exists()

        content = readme_path.read_text(encoding="utf-8")
        assert "TNFR Presets" in content
        assert "resonant_bootstrap" in content

    def test_yaml_presets_are_valid_yaml(self):
        """Test that YAML preset files are valid YAML."""
        pytest.importorskip("yaml")
        import yaml

        presets_dir = Path("presets")
        yaml_files = list(presets_dir.glob("*.yaml"))
        assert len(yaml_files) > 0

        for yaml_file in yaml_files:
            content = yaml_file.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            assert data is not None, f"{yaml_file.name} should contain valid YAML"
            assert isinstance(data, dict), f"{yaml_file.name} should be a YAML dict"

    def test_yaml_presets_have_required_fields(self):
        """Test that YAML presets have required metadata fields."""
        pytest.importorskip("yaml")
        import yaml

        presets_dir = Path("presets")
        yaml_files = [f for f in presets_dir.glob("*.yaml") if f.name != "README.md"]

        for yaml_file in yaml_files:
            content = yaml_file.read_text(encoding="utf-8")
            data = yaml.safe_load(content)

            # Check required top-level keys
            assert "metadata" in data, f"{yaml_file.name} should have metadata"
            assert "sequence" in data, f"{yaml_file.name} should have sequence"

            metadata = data["metadata"]
            assert "name" in metadata, f"{yaml_file.name} metadata should have name"
            assert "description" in metadata, f"{yaml_file.name} metadata should have description"
            assert "operators" in metadata, f"{yaml_file.name} metadata should have operators"


class TestCLIStdoutStderrCapture:
    """Tests specifically for stdout/stderr capture as requested."""

    def test_version_output_captured(self, capsys):
        """Test that --version output is captured correctly."""
        rc = main(["--version"])
        assert rc == 0

        captured = capsys.readouterr()
        assert captured.out.strip() != ""
        assert captured.err == ""  # Version should go to stdout

    def test_error_messages_captured(self, capsys):
        """Test that error messages are captured."""
        rc = main(["run", "--preset", "nonexistent"])
        assert rc == 1

        captured = capsys.readouterr()
        assert "Unknown preset" in captured.out

    def test_math_run_output_captured(self, capsys):
        """Test that math.run output is captured."""
        rc = main([
            "math.run",
            "--nodes", "3",
            "--steps", "1",
        ])
        assert rc == 0

        captured = capsys.readouterr()
        # Should have some output
        assert len(captured.out) > 0

    def test_epi_validate_output_captured(self, capsys):
        """Test that epi.validate output is captured."""
        rc = main([
            "epi.validate",
            "--nodes", "3",
            "--steps", "1",
        ])
        assert rc == 0

        captured = capsys.readouterr()
        assert "[EPI.VALIDATE]" in captured.out
        assert "Validation Summary" in captured.out

    def test_help_output_captured_for_all_subcommands(self, capsys):
        """Test that help is captured for all subcommands."""
        subcommands = ["run", "math.run", "epi.validate", "sequence", "metrics"]

        for subcmd in subcommands:
            with pytest.raises(SystemExit) as exc_info:
                main([subcmd, "--help"])
            assert exc_info.value.code == 0

            captured = capsys.readouterr()
            assert len(captured.out) > 0, f"{subcmd} --help should produce output"
            assert "usage:" in captured.out.lower()


class TestCLIIntegrationScenarios:
    """Integration tests for realistic CLI usage scenarios."""

    def test_run_preset_with_yaml_file(self, tmp_path, capsys):
        """Test running a simulation from a YAML sequence file."""
        # Create a minimal YAML sequence file
        yaml_content = """
sequence:
  - AL
  - EN
  - WAIT: 1
"""
        yaml_file = tmp_path / "test_sequence.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        rc = main([
            "sequence",
            "--sequence-file", str(yaml_file),
            "--nodes", "3",
        ])
        assert rc == 0

    def test_math_run_exports_history(self, tmp_path, capsys):
        """Test that math.run can export history."""
        history_file = tmp_path / "history.json"

        rc = main([
            "math.run",
            "--nodes", "3",
            "--steps", "2",
            "--save-history", str(history_file),
        ])
        assert rc == 0

        # Check that history file was created
        assert history_file.exists()

        # Verify it's valid JSON
        with open(history_file, encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict)

    def test_epi_validate_different_topologies(self, capsys):
        """Test epi.validate with different network topologies."""
        topologies = ["ring", "complete"]

        for topo in topologies:
            rc = main([
                "epi.validate",
                "--nodes", "4",
                "--topology", topo,
                "--steps", "2",
            ])
            assert rc == 0, f"Validation should pass for {topo} topology"

            captured = capsys.readouterr()
            assert "[EPI.VALIDATE]" in captured.out

    def test_metrics_with_math_and_epi(self, tmp_path, capsys):
        """Test that metrics command still works alongside new commands."""
        metrics_file = tmp_path / "metrics.json"

        rc = main([
            "metrics",
            "--nodes", "3",
            "--steps", "5",
            "--save", str(metrics_file),
        ])
        assert rc == 0

        assert metrics_file.exists()

        with open(metrics_file, encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict)
