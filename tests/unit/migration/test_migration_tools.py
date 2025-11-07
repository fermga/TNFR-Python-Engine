"""Tests for migration tools."""

import pytest
from tools.migration.migration_checker import (
    MigrationChecker,
    MigrationReport,
    MigrationIssue,
    IssueLevel,
)
from tools.migration.sequence_upgrader import (
    SequenceUpgrader,
    UpgradeResult,
)


class TestMigrationChecker:
    """Tests for MigrationChecker."""
    
    def test_check_sequence_thol_without_destabilizer(self):
        """Test detection of THOL without destabilizer."""
        checker = MigrationChecker()
        sequence = ["emission", "reception", "self_organization"]
        
        issues = checker.check_sequence(sequence)
        
        assert len(issues) > 0
        assert any(
            issue.level == IssueLevel.ERROR and "destabilizer" in issue.message.lower()
            for issue in issues
        )
    
    def test_check_sequence_thol_with_destabilizer(self):
        """Test THOL with destabilizer passes."""
        checker = MigrationChecker()
        sequence = ["emission", "dissonance", "self_organization"]
        
        issues = checker.check_sequence(sequence)
        
        # Should not have THOL error
        assert not any(
            issue.level == IssueLevel.ERROR and "destabilizer" in issue.message.lower()
            for issue in issues
        )
    
    def test_check_sequence_frequency_jump(self):
        """Test detection of frequency jumps."""
        checker = MigrationChecker()
        sequence = ["silence", "emission"]
        
        issues = checker.check_sequence(sequence)
        
        assert any(
            issue.level == IssueLevel.WARNING and "frequency" in issue.message.lower()
            for issue in issues
        )
    
    def test_scan_sequences(self):
        """Test scanning multiple sequences."""
        checker = MigrationChecker()
        sequences = [
            ["emission", "coherence"],
            ["emission", "reception", "self_organization"],
            ["silence", "emission"],
        ]
        
        report = checker.scan_sequences(sequences)
        
        assert report.sequences_checked == 3
        assert len(report.issues) > 0
        assert report.has_errors  # THOL issue
    
    def test_migration_report_summary(self):
        """Test report summary generation."""
        report = MigrationReport()
        report.sequences_checked = 2
        report.issues.append(
            MigrationIssue(
                level=IssueLevel.ERROR,
                message="Test error",
                suggestion="Fix it"
            )
        )
        
        summary = report.summary()
        
        assert "Test error" in summary
        assert "Fix it" in summary
        assert "Errors: 1" in summary


class TestSequenceUpgrader:
    """Tests for SequenceUpgrader."""
    
    def test_upgrade_thol_issue(self):
        """Test fixing THOL validation issue."""
        upgrader = SequenceUpgrader()
        sequence = ["emission", "reception", "self_organization"]
        
        result = upgrader.upgrade_sequence(sequence)
        
        assert result.was_upgraded
        assert "dissonance" in [op.lower() for op in result.upgraded_sequence]
        assert any("DISSONANCE" in imp for imp in result.improvements)
    
    def test_upgrade_frequency_transition(self):
        """Test fixing frequency transitions."""
        upgrader = SequenceUpgrader()
        sequence = ["silence", "emission"]
        
        result = upgrader.upgrade_sequence(sequence, preserve_length=False)
        
        # Should insert medium frequency operator
        assert len(result.upgraded_sequence) > len(result.original_sequence)
        assert any("transition" in imp.lower() for imp in result.improvements)
    
    def test_upgrade_preserve_length(self):
        """Test upgrade with length preservation."""
        upgrader = SequenceUpgrader()
        sequence = ["emission", "reception", "self_organization"]
        
        result = upgrader.upgrade_sequence(sequence, preserve_length=True)
        
        # Should replace operator, not insert
        assert len(result.upgraded_sequence) == len(result.original_sequence)
    
    def test_upgrade_balance_operators(self):
        """Test balancing stabilizers and destabilizers."""
        upgrader = SequenceUpgrader()
        sequence = ["emission", "dissonance"]
        
        result = upgrader.upgrade_sequence(sequence, preserve_length=False)
        
        # Should add stabilizer
        assert "coherence" in [op.lower() for op in result.upgraded_sequence]
        assert any("coherence" in imp.lower() for imp in result.improvements)
    
    def test_improve_to_target(self):
        """Test iterative improvement to target health."""
        upgrader = SequenceUpgrader(target_health=0.70)
        sequence = ["emission", "reception"]
        
        result = upgrader.improve_to_target(sequence, max_iterations=3)
        
        # Should have made improvements
        assert len(result.upgraded_sequence) >= len(result.original_sequence)
        assert len(result.improvements) > 0
    
    def test_upgrade_result_summary(self):
        """Test upgrade result summary."""
        result = UpgradeResult(
            original_sequence=["emission", "reception"],
            upgraded_sequence=["emission", "reception", "coherence"],
            improvements=["Added COHERENCE"],
            original_health=0.5,
            upgraded_health=0.75
        )
        
        summary = result.summary()
        
        assert "emission" in summary
        assert "coherence" in summary
        assert "Added COHERENCE" in summary
        assert "0.50" in summary
        assert "0.75" in summary


class TestIntegration:
    """Integration tests for migration tools."""
    
    def test_checker_and_upgrader_integration(self):
        """Test using checker to find issues, then upgrader to fix them."""
        # Step 1: Check for issues
        checker = MigrationChecker()
        sequence = ["emission", "reception", "self_organization"]
        issues = checker.check_sequence(sequence)
        
        assert any(issue.level == IssueLevel.ERROR for issue in issues)
        
        # Step 2: Upgrade to fix issues
        upgrader = SequenceUpgrader()
        result = upgrader.upgrade_sequence(sequence)
        
        # Step 3: Check upgraded sequence
        upgraded_issues = checker.check_sequence(result.upgraded_sequence)
        
        # Should have fewer or no errors
        upgraded_errors = [i for i in upgraded_issues if i.level == IssueLevel.ERROR]
        original_errors = [i for i in issues if i.level == IssueLevel.ERROR]
        assert len(upgraded_errors) <= len(original_errors)
    
    def test_multiple_sequences_workflow(self):
        """Test typical workflow with multiple sequences."""
        sequences = [
            ["emission", "coherence"],
            ["emission", "reception", "self_organization"],
            ["silence", "emission"],
        ]
        
        # Check all sequences
        checker = MigrationChecker()
        report = checker.scan_sequences(sequences)
        
        assert report.sequences_checked == 3
        
        # Upgrade problematic sequences
        upgrader = SequenceUpgrader()
        upgraded_sequences = []
        
        for seq in sequences:
            result = upgrader.upgrade_sequence(seq)
            upgraded_sequences.append(result.upgraded_sequence)
        
        # Verify upgrades improved things
        upgraded_report = checker.scan_sequences(upgraded_sequences)
        
        # Should have fewer or same number of errors
        assert len([i for i in upgraded_report.issues if i.level == IssueLevel.ERROR]) <= \
               len([i for i in report.issues if i.level == IssueLevel.ERROR])
