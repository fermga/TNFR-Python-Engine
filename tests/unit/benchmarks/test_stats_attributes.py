"""Tests for pstats.Stats attribute usage in benchmarks.

These tests validate that benchmarks correctly use the documented attributes
of pstats.Stats objects, particularly the `.stats` dictionary attribute.
"""

import cProfile
import pstats
from pathlib import Path

import pytest


class TestPstatsStatsAttributes:
    """Test that pstats.Stats objects have expected attributes."""

    def test_pstats_stats_has_stats_attribute(self):
        """pstats.Stats must have a .stats attribute (documented behavior)."""
        profile = cProfile.Profile()
        profile.enable()
        _ = sum(range(100))
        profile.disable()
        
        stats_obj = pstats.Stats(profile)
        
        # The .stats attribute should exist
        assert hasattr(stats_obj, "stats")
        assert isinstance(stats_obj.stats, dict)

    def test_pstats_stats_dict_structure(self):
        """The .stats dict should have the expected structure."""
        profile = cProfile.Profile()
        profile.enable()
        _ = sum(range(100))
        profile.disable()
        
        stats_obj = pstats.Stats(profile)
        
        # Each entry should be (filename, lineno, func) -> (stats_tuple)
        for key, value in stats_obj.stats.items():
            assert isinstance(key, tuple)
            assert len(key) == 3  # (filename, lineno, func)
            
            assert isinstance(value, tuple)
            assert len(value) == 5  # (cc, nc, tt, ct, callers)

    def test_stats_iteration_pattern_used_in_benchmarks(self):
        """Validate the iteration pattern used in compute_si_profile.py."""
        profile = cProfile.Profile()
        profile.enable()
        
        def test_function():
            return sum(range(100))
        
        result = test_function()
        profile.disable()
        
        stats_obj = pstats.Stats(profile)
        
        # This is the pattern used in benchmarks/compute_si_profile.py:112
        rows = []
        for (filename, lineno, func), (cc, nc, tt, ct, callers) in stats_obj.stats.items():
            rows.append({
                "function": func,
                "file": filename,
                "line": lineno,
                "callcount": cc,
                "reccallcount": nc,
                "totaltime": tt,
                "cumtime": ct,
            })
        
        # Should have collected some rows
        assert len(rows) > 0
        
        # Verify structure
        for row in rows:
            assert "function" in row
            assert "file" in row
            assert "line" in row
            assert "callcount" in row
            assert "reccallcount" in row
            assert "totaltime" in row
            assert "cumtime" in row

    def test_stats_has_other_documented_methods(self):
        """Verify pstats.Stats has other expected methods."""
        profile = cProfile.Profile()
        profile.enable()
        _ = sum(range(100))
        profile.disable()
        
        stats_obj = pstats.Stats(profile)
        
        # Documented methods that benchmarks might use
        assert hasattr(stats_obj, "sort_stats")
        assert hasattr(stats_obj, "dump_stats")
        assert hasattr(stats_obj, "print_stats")
        assert callable(stats_obj.sort_stats)
        assert callable(stats_obj.dump_stats)


class TestBenchmarkStatsUsagePatterns:
    """Test safe patterns for using Stats objects in benchmarks."""

    def test_safe_stats_extraction_pattern(self):
        """Demonstrate safe pattern for extracting stats data."""
        profile = cProfile.Profile()
        profile.enable()
        
        def sample_func():
            return [i ** 2 for i in range(50)]
        
        result = sample_func()
        profile.disable()
        
        stats_obj = pstats.Stats(profile)
        
        # Safe pattern: extract specific data
        def extract_function_stats(stats, target_func_name):
            """Extract stats for a specific function."""
            results = []
            for (filename, lineno, func), (cc, nc, tt, ct, callers) in stats.stats.items():
                if func == target_func_name:
                    results.append({
                        "callcount": cc,
                        "totaltime": tt,
                        "cumtime": ct,
                    })
            return results
        
        # This should work without errors
        sample_stats = extract_function_stats(stats_obj, "sample_func")
        assert isinstance(sample_stats, list)

    def test_stats_can_be_sorted(self):
        """Stats objects support sorting before accessing .stats."""
        profile = cProfile.Profile()
        profile.enable()
        _ = sum(range(100))
        profile.disable()
        
        stats_obj = pstats.Stats(profile)
        
        # Sort by cumulative time (common pattern in benchmarks)
        stats_obj.sort_stats("cumtime")
        
        # After sorting, .stats should still be accessible
        assert hasattr(stats_obj, "stats")
        assert isinstance(stats_obj.stats, dict)


class TestStatsDocumentation:
    """Document the expected behavior of Stats objects for benchmarks."""

    def test_stats_attribute_is_documented(self):
        """The .stats attribute is part of pstats.Stats public API.
        
        From Python docs:
        The .stats attribute is a dictionary mapping function info to 
        (cc, nc, tt, ct, callers) where:
        - cc: call count
        - nc: number of recursive calls
        - tt: total time spent in function
        - ct: cumulative time spent in function and subfunctions
        - callers: dictionary of calling functions
        """
        profile = cProfile.Profile()
        profile.enable()
        _ = sum(range(10))
        profile.disable()
        
        stats_obj = pstats.Stats(profile)
        
        # Document what .stats contains
        for key, value in list(stats_obj.stats.items())[:1]:
            filename, lineno, func = key
            cc, nc, tt, ct, callers = value
            
            assert isinstance(filename, str)
            assert isinstance(lineno, int)
            assert isinstance(func, str)
            assert isinstance(cc, int)
            assert isinstance(nc, int)
            assert isinstance(tt, (int, float))
            assert isinstance(ct, (int, float))
            assert isinstance(callers, dict)
