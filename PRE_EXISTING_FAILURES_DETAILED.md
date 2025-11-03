# Complete List of Pre-Existing Test Failures

## Summary

Total failing/error tests: 130

See [PRE_EXISTING_FAILURES.md](./PRE_EXISTING_FAILURES.md) for categorized analysis and recommended fixes.

## Complete List

```
1. FAILED tests/golden/test_classic_snapshots.py::test_classic_runtime_sequence_matches_golden_snapshot - assert 0.10000000000000003 == -0.2615625 ± 2.6e-07
2. FAILED tests/integration/test_cli.py::test_cli_sequence_handles_deeply_nested_blocks - tnfr.operators.grammar.TholClosureError: self_organization block requires silence closure
3. FAILED tests/integration/test_cli.py::test_cli_without_history_args[sequence] - tnfr.operators.grammar.MutationPreconditionError: mutation mutation requires dissonance within window 3
4. FAILED tests/integration/test_docs_fase2_integration.py::test_fase2_integration_doc_executes - FileNotFoundError: [Errno 2] No such file or directory: 'docs/fase2_integration.md'
5. FAILED tests/integration/test_program.py::test_play_handles_deeply_nested_blocks - tnfr.operators.grammar.TholClosureError: self_organization block requires contraction closure
6. FAILED tests/integration/test_program.py::test_flatten_plain_sequence_skips_materialization - assert True is False
7. FAILED tests/integration/test_public_api.py::test_public_exports - AssertionError: assert {'__version__..._hz_str', ...} == {'__version__...'run', 'step'}
8. FAILED tests/integration/test_run_sequence_critical_paths.py::test_run_sequence_mixed_operation_types - ValueError: unknown glyph: Glyph.THOL
9. FAILED tests/integration/test_run_sequence_trajectories.py::test_run_sequence_target_all_nodes - ValueError: unknown glyph: Glyph.SHA
10. FAILED tests/integration/test_sense_index_parallel.py::test_parallel_si_matches_sequential_for_large_graph - AssertionError: parallel path should instantiate the executor
11. FAILED tests/integration/test_validation_rules.py::test_check_oz_to_zhir_requires_recent_oz - tnfr.operators.grammar.MutationPreconditionError: mutation mutation requires dissonance within window 3
12. FAILED tests/math_integration/test_operators_wiring.py::test_node_accepts_direct_operator_instances - ValueError: Invalid sequence: missing reception→coherence segment
13. FAILED tests/math_integration/test_operators_wiring.py::test_node_constructs_operators_from_factory_parameters - AssertionError: assert 'coherence_expectation' in {'frequency_enforced': True, 'frequency_expectation': 0.25795514448650986, 'frequency_positive': True, 'frequency_projection_passed': True, ...}
14. FAILED tests/math_integration/test_projection.py::test_run_sequence_with_validation_supports_keyword_only_projector - ValueError: Invalid sequence: empty sequence
15. FAILED tests/property/test_initialization_properties.py::test_init_node_attrs_respects_graph_configuration - ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
16. FAILED tests/property/test_structured_io_roundtrip.py::test_structured_file_roundtrip[.yaml-_write_yaml] - AssertionError: assert '1e-05' == 1e-05
17. FAILED tests/stress/test_program_trace.py::test_program_trace_rotates_without_dropping_thol_history - ValueError: unknown glyph: Glyph.REMESH
18. FAILED tests/unit/dynamics/test_canon.py::test_validate_canon_clamps - TypeError: '<=' not supported between instances of 'float' and 'dict'
19. FAILED tests/unit/dynamics/test_dnfr_cache.py::test_prepare_dnfr_data_uses_public_cache_factory - assert 0 >= 1
20. FAILED tests/unit/dynamics/test_dnfr_cache.py::test_neighbor_sum_buffers_reused_and_results_stable[False] - AssertionError: assert array([0.95533649, 1.88921818, 1.7806721 , 1.63213597, 0.82533561]) is None
21. FAILED tests/unit/dynamics/test_dnfr_cache.py::test_cached_nodes_and_A_returns_none_without_numpy - assert array([[0., 1.],\n       [1., 0.]]) is None
22. FAILED tests/unit/dynamics/test_dnfr_cache.py::test_cached_nodes_and_A_requires_numpy - Failed: DID NOT RAISE <class 'RuntimeError'>
23. FAILED tests/unit/dynamics/test_dnfr_neighbor_means.py::test_neighbor_mean_zero_vector_length_preserves_theta - assert 0.0 > 0
24. FAILED tests/unit/dynamics/test_dnfr_parallel_chunks.py::test_parallel_chunks_cover_all_nodes_once - AssertionError: every node scheduled exactly once
25. FAILED tests/unit/dynamics/test_dynamics_helpers.py::test_init_and_refresh_dnfr_cache - assert 0.0 == 0.1 ± 1.0e-07
26. FAILED tests/unit/dynamics/test_dynamics_helpers.py::test_refresh_dnfr_vectors_python_fallback - assert [0.0, 0.0, 0.0] == approx([0.0 ±....4 ± 4.0e-07])
27. FAILED tests/unit/dynamics/test_dynamics_helpers.py::test_prepare_dnfr_passes_configured_jobs - KeyError: 'n_jobs'
28. FAILED tests/unit/dynamics/test_dynamics_helpers.py::test_prepare_dnfr_falls_back_to_metrics_compute_si - KeyError: 'call'
29. FAILED tests/unit/dynamics/test_dynamics_helpers.py::test_broadcast_accumulator_matches_legacy_and_speed - assert 0.0013757639999312232 <= (0.0011272120000285213 * 1.1)
30. FAILED tests/unit/dynamics/test_dynamics_run.py::test_step_respects_n_jobs_overrides - AssertionError: assert {'dnfr': 3, '...': 6, 'vf': 7} == {'dnfr': 3, '... 'si': 4, ...}
31. FAILED tests/unit/dynamics/test_dynamics_run.py::test_step_defaults_to_graph_jobs - AssertionError: assert {'dnfr': 2, '...5, 'vf': None} == {'dnfr': 2, '... 'si': 3, ...}
32. FAILED tests/unit/dynamics/test_dynamics_run.py::test_update_nodes_clamps_out_of_range_values - assert 0.0 == 1.0 ± 1.0e-06
33. FAILED tests/unit/dynamics/test_dynamics_vectorized.py::test_vectorization_falls_back_without_numpy - AssertionError: assert array([0.1, 0.2, 0.3, 0.4, 0.5]) is None
34. FAILED tests/unit/dynamics/test_dynamics_vectorized.py::test_compute_dnfr_auto_vectorizes_when_numpy_present - assert ([])
35. FAILED tests/unit/dynamics/test_dynamics_vectorized.py::test_prepare_dnfr_data_skips_degree_when_topology_disabled - AssertionError: assert {0: 1.0, 1: 2.0, 2: 2.0, 3: 2.0, ...} is None
36. FAILED tests/unit/dynamics/test_dynamics_vectorized.py::test_vectorized_matches_python_and_is_faster_large_graph - assert 0.01579262200004905 < 0.015204304999997476
37. FAILED tests/unit/dynamics/test_gamma.py::test_gamma_spec_normalized_once - AttributeError: 'module' object at tnfr.utils.graph has no attribute 'graph'
38. FAILED tests/unit/dynamics/test_gamma.py::test_kuramoto_cache_step_limit - assert 0 == 2
39. FAILED tests/unit/dynamics/test_glyph_selector_parallel.py::test_default_selector_parallel_matches_sequential - _pickle.PicklingError: Can't pickle <function _glyph_proposal_worker at 0x7f1cbec945e0>: it's not the same object as tnfr.dynamics.selectors._glyph_proposal_worker
40. FAILED tests/unit/dynamics/test_glyph_selector_parallel.py::test_param_selector_parallel_matches_sequential - _pickle.PicklingError: Can't pickle <function _param_base_worker at 0x7f1cbec94360>: it's not the same object as tnfr.dynamics.selectors._param_base_worker
41. FAILED tests/unit/dynamics/test_glyph_selector_parallel.py::test_parallel_selector_is_deterministic - _pickle.PicklingError: Can't pickle <function _param_base_worker at 0x7f1cbec94360>: it's not the same object as tnfr.dynamics.selectors._param_base_worker
42. FAILED tests/unit/dynamics/test_glyph_selector_parallel.py::test_parallel_respects_since_counters - _pickle.PicklingError: Can't pickle <function _glyph_proposal_worker at 0x7f1cbec945e0>: it's not the same object as tnfr.dynamics.selectors._glyph_proposal_worker
43. FAILED tests/unit/dynamics/test_glyph_selector_parallel.py::test_parallel_canonical_hooks_order - _pickle.PicklingError: Can't pickle <function _param_base_worker at 0x7f1cbec94360>: it's not the same object as tnfr.dynamics.selectors._param_base_worker
44. FAILED tests/unit/dynamics/test_grammar.py::test_precondition_oz_to_zhir - tnfr.operators.grammar.MutationPreconditionError: mutation mutation requires dissonance within window 3
45. FAILED tests/unit/dynamics/test_grammar.py::test_thol_closure - tnfr.operators.grammar.TholClosureError: self_organization block requires silence closure
46. FAILED tests/unit/dynamics/test_grammar.py::test_repeat_invalid_fallback_string - Failed: DID NOT RAISE <class 'tnfr.operators.grammar.RepeatWindowError'>
47. FAILED tests/unit/dynamics/test_grammar.py::test_repeat_invalid_fallback_type - tnfr.operators.grammar.GrammarConfigurationError: invalid cfg_soft configuration: fallbacks.ZHIR: <object object at 0x7f1cb9fc10f0> is not of type 'string'
48. FAILED tests/unit/dynamics/test_grammar.py::test_choose_glyph_records_violation - tnfr.operators.grammar.MutationPreconditionError: mutation mutation requires dissonance within window 3
49. FAILED tests/unit/dynamics/test_grammar.py::test_apply_glyph_with_grammar_records_violation - tnfr.operators.grammar.MutationPreconditionError: mutation mutation requires dissonance within window 3
50. FAILED tests/unit/dynamics/test_grammar.py::test_apply_glyph_with_grammar_equivalence - tnfr.operators.grammar.MutationPreconditionError: mutation mutation requires dissonance within window 3
51. FAILED tests/unit/dynamics/test_grammar.py::test_apply_glyph_with_grammar_multiple_nodes - ValueError: unknown glyph: Glyph.ZHIR
52. FAILED tests/unit/dynamics/test_grammar.py::test_apply_glyph_with_grammar_accepts_iterables - ValueError: unknown glyph: Glyph.ZHIR
53. FAILED tests/unit/dynamics/test_grammar.py::test_apply_glyph_with_grammar_defaults_window_from_graph - AttributeError: 'tnfr.operators' has no attribute 'apply_glyph'
54. FAILED tests/unit/dynamics/test_integrators.py::test_epi_limits_preserved[euler] - AssertionError: assert {'continuous'...': (0.0, 1.0)} == 1.0 ± 1.0e-06
55. FAILED tests/unit/dynamics/test_integrators.py::test_epi_limits_preserved[rk4] - AssertionError: assert {'continuous'...': (0.0, 1.0)} == 1.0 ± 1.0e-06
56. FAILED tests/unit/dynamics/test_nav.py::test_nav_random_applies_jitter - KeyError: 'magnitude'
57. FAILED tests/unit/dynamics/test_nav.py::test_nav_random_negative_dnfr_base - assert -0.38640718867047996 == -0.577 ± 5.8e-07
58. FAILED tests/unit/dynamics/test_neighbor_accumulation_vectorized.py::test_broadcast_accumulator_degree_totals_without_chunking - assert 6 == 34
59. FAILED tests/unit/dynamics/test_operators.py::test_dissonance_operator_increases_dnfr_and_tracks_phase - ValueError: Invalid sequence: must start with emission, recursivity
60. FAILED tests/unit/dynamics/test_operators.py::test_random_jitter_missing_nodenx - Failed: DID NOT RAISE <class 'ImportError'>
61. FAILED tests/unit/dynamics/test_operators.py::test_rng_cache_disabled_with_size_zero - assert 0.44460211902766866 == -0.1014695360860901
62. FAILED tests/unit/dynamics/test_runtime_clamps.py::test_apply_canonical_clamps_clamps_and_logs - assert 0.0 == 1.0 ± 1.0e-06
63. FAILED tests/unit/dynamics/test_runtime_clamps.py::test_apply_canonical_clamps_updates_mapping_without_graph - assert 0.0 == -1.0 ± 1.0e-06
64. FAILED tests/unit/dynamics/test_runtime_clamps.py::test_apply_canonical_clamps_respects_disabled_theta_wrap - assert 0.0 == 1.0 ± 1.0e-06
65. FAILED tests/unit/dynamics/test_vf_coherence.py::test_adapt_vf_clamps_to_bounds[python] - AssertionError: parallel adaptation must submit work
66. FAILED tests/unit/metrics/test_coherence_cache.py::test_use_numpy_parameter_matches_loops - _pickle.PicklingError: Can't pickle <function _parallel_wij_worker at 0x7f1cbec1d8a0>: it's not the same object as tnfr.metrics.coherence._parallel_wij_worker
67. FAILED tests/unit/metrics/test_coherence_cache.py::test_coherence_parallel_consistency_neighbors_and_all - _pickle.PicklingError: Can't pickle <function _parallel_wij_worker at 0x7f1cbec1d8a0>: it's not the same object as tnfr.metrics.coherence._parallel_wij_worker
68. FAILED tests/unit/metrics/test_collect_selector_metrics.py::test_collect_selector_metrics_process_pool - KeyError: 'max_workers'
69. FAILED tests/unit/metrics/test_compute_Si_numpy_usage.py::test_compute_Si_vectorized_uses_bulk_helper - assert 0 == 1
70. FAILED tests/unit/metrics/test_compute_Si_numpy_usage.py::test_compute_Si_vectorized_buffer_reuse_matches_python - assert []
71. FAILED tests/unit/metrics/test_compute_Si_numpy_usage.py::test_compute_Si_reads_jobs_from_graph - assert [] == [3]
72. FAILED tests/unit/metrics/test_compute_Si_numpy_usage.py::test_compute_Si_vectorized_respects_chunk_size - assert ([])
73. FAILED tests/unit/metrics/test_compute_Si_numpy_usage.py::test_compute_Si_vectorized_chunked_results_match - assert ([])
74. FAILED tests/unit/metrics/test_compute_Si_numpy_usage.py::test_compute_Si_vectorized_skips_isolated_nodes - assert []
75. FAILED tests/unit/metrics/test_compute_Si_numpy_usage.py::test_compute_Si_python_parallel_chunk_size - assert []
76. FAILED tests/unit/metrics/test_compute_Si_numpy_usage.py::test_compute_Si_edge_indices_cache_invalidation - assert ([])
77. FAILED tests/unit/metrics/test_diagnosis_parallel.py::test_parallel_diagnosis_matches_serial[3] - _pickle.PicklingError: Can't pickle <function _diagnosis_worker_chunk at 0x7f1cbec1fce0>: it's not the same object as tnfr.metrics.diagnosis._diagnosis_worker_chunk
78. FAILED tests/unit/metrics/test_diagnosis_parallel.py::test_diagnosis_vectorized_matches_python - _pickle.PicklingError: Can't pickle <function _diagnosis_worker_chunk at 0x7f1cbec1fce0>: it's not the same object as tnfr.metrics.diagnosis._diagnosis_worker_chunk
79. FAILED tests/unit/metrics/test_diagnosis_parallel.py::test_diagnosis_python_parallel_without_numpy - _pickle.PicklingError: Can't pickle <function _diagnosis_worker_chunk at 0x7f1cbec1fce0>: it's not the same object as tnfr.metrics.diagnosis._diagnosis_worker_chunk
80. FAILED tests/unit/metrics/test_invariants.py::test_clamps_numeric_stability - TypeError: float() argument must be a string or a real number, not 'dict'
81. FAILED tests/unit/metrics/test_metrics.py::test_track_stability_parallel_fallback - assert ([])
82. FAILED tests/unit/metrics/test_metrics.py::test_update_sigma_uses_default_window - KeyError: 'window'
83. FAILED tests/unit/metrics/test_metrics.py::test_metrics_detailed_verbosity_runs_collectors - AssertionError: assert [] == ['phase', 'si..., 'aggregate']
84. FAILED tests/unit/metrics/test_metrics.py::test_register_metrics_callbacks_respects_verbosity - AssertionError: assert [] == ['coherence']
85. FAILED tests/unit/metrics/test_metrics.py::test_aggregate_si_python_parallel - assert ([])
86. FAILED tests/unit/metrics/test_metrics.py::test_build_metrics_summary_reuses_metrics_helpers - AttributeError: 'object' object has no attribute 'graph'
87. FAILED tests/unit/metrics/test_metrics.py::test_build_metrics_summary_handles_empty_latency - AttributeError: 'object' object has no attribute 'graph'
88. FAILED tests/unit/metrics/test_metrics.py::test_build_metrics_summary_accepts_unbounded_limit - AttributeError: 'object' object has no attribute 'graph'
89. FAILED tests/unit/metrics/test_metrics.py::test_advanced_metrics_process_pool_fallback - assert 0 == 2
90. FAILED tests/unit/metrics/test_neighbor_phase_mean_missing_trig.py::test_neighbor_phase_mean_list_delegates_generic - assert 0.0 == 1.23 ± 1.2e-06
91. FAILED tests/unit/metrics/test_neighbor_phase_mean_no_graph.py::test_neighbor_phase_mean_uses_generic - assert 0 == 1
92. FAILED tests/unit/metrics/test_trig_cache_reuse.py::test_trig_cache_reuse_between_modules - assert 4 == 2
93. FAILED tests/unit/metrics/test_trig_cache_reuse.py::test_trig_cache_rebuilds_after_direct_theta_mutation - assert 2 == 4
94. FAILED tests/unit/operators/test_grammar_module.py::test_enforce_canonical_grammar_respects_thol_state - tnfr.operators.grammar.TholClosureError: self_organization block requires silence closure
95. FAILED tests/unit/operators/test_grammar_module.py::test_enforce_canonical_grammar_accepts_canonical_strings - tnfr.operators.grammar.TholClosureError: self_organization block requires silence closure
96. FAILED tests/unit/operators/test_grammar_module.py::test_mutation_precondition_error_uses_structural_order - tnfr.operators.grammar.MutationPreconditionError: mutation mutation requires dissonance within window 3
97. FAILED tests/unit/operators/test_grammar_module.py::test_thol_closure_error_uses_structural_order - tnfr.operators.grammar.TholClosureError: self_organization block requires silence closure
98. FAILED tests/unit/operators/test_grammar_module.py::test_apply_glyph_with_grammar_accepts_glyph_instances - ValueError: unknown glyph: Glyph.AL
99. FAILED tests/unit/operators/test_grammar_module.py::test_apply_glyph_with_grammar_translates_canonical_strings - ValueError: unknown glyph: Glyph.AL
100. FAILED tests/unit/operators/test_grammar_module.py::test_repeat_window_error_uses_structural_names - tnfr.operators.grammar.RepeatWindowError: emission repeats within window 2
101. FAILED tests/unit/operators/test_grammar_module.py::test_apply_glyph_with_grammar_canonical_string_violation - tnfr.operators.grammar.TholClosureError: self_organization block requires silence closure
102. FAILED tests/unit/structural/test_bepi_node_and_validators.py::test_nodenx_epi_roundtrip_serializes_bepi - TypeError: Unsupported BEPI value type: <class 'tnfr.mathematics.epi.BEPIElement'>
103. FAILED tests/unit/structural/test_bepi_node_and_validators.py::test_graph_validators_accept_bepi_payload - TypeError: Unsupported BEPI value type: <class 'tnfr.mathematics.epi.BEPIElement'>
104. FAILED tests/unit/structural/test_bepi_node_and_validators.py::test_runtime_validator_clamps_bepi_components - TypeError: Unsupported BEPI value type: <class 'tnfr.mathematics.epi.BEPIElement'>
105. FAILED tests/unit/structural/test_config_apply.py::test_load_config_accepts_mapping - AssertionError: assert {} == {'RANDOM_SEED': 1}
106. FAILED tests/unit/structural/test_config_apply.py::test_load_config_requires_object - Failed: DID NOT RAISE <class 'ValueError'>
107. FAILED tests/unit/structural/test_config_apply.py::test_apply_config_passes_path_object - KeyError: 'path'
108. FAILED tests/unit/structural/test_kahan_sum.py::test_kahan_sum_nd_compensates_cancellation_1d - assert 1.0 == 0.0
109. FAILED tests/unit/structural/test_logging_module.py::test_get_logger_configures_root_once - ImportError: module tnfr.utils.init not in sys.modules
110. FAILED tests/unit/structural/test_logging_threadsafe.py::test_get_logger_threadsafe - ImportError: module tnfr.utils.init not in sys.modules
111. FAILED tests/unit/structural/test_logging_threadsafe.py::test_get_logger_preserves_existing_level - ImportError: module tnfr.utils.init not in sys.modules
112. FAILED tests/unit/structural/test_logging_threadsafe.py::test_get_logger_sets_level_when_notset - ImportError: module tnfr.utils.init not in sys.modules
113. FAILED tests/unit/structural/test_logging_threadsafe.py::test_get_logger_multiple_calls_do_not_reconfigure_root - ImportError: module tnfr.utils.init not in sys.modules
114. FAILED tests/unit/structural/test_node_set_checksum.py::test_increment_edge_version_clears_node_repr_cache - assert 0 == 3
115. FAILED tests/unit/structural/test_observers.py::test_phase_sync_and_kuramoto_order_share_metrics - assert 0 == 1
116. FAILED tests/unit/structural/test_observers.py::test_glyph_load_uses_module_constants - assert 0.0 == 0.5 ± 5.0e-07
117. FAILED tests/unit/structural/test_observers.py::test_wbar_uses_default_window - assert 1.0 == 1.25 ± 1.2e-06
118. FAILED tests/unit/structural/test_prepare_network.py::test_prepare_network_records_callback_error_when_observer_missing - KeyError: '_callback_errors'
119. FAILED tests/unit/structural/test_prepare_network.py::test_prepare_network_attaches_standard_observer - assert [] == [<networkx.cl...7f1cb778c4a0>]
120. FAILED tests/unit/structural/test_rng_base_seed.py::test_seed_hash_metrics - assert (24 - 24) == 1
121. FAILED tests/unit/structural/test_rng_base_seed.py::test_seed_hash_evictions_recorded - assert (0 - 0) == 1
122. FAILED tests/unit/structural/test_safe_write.py::test_safe_write_binary_mode[True] - AttributeError: 'module' object at tnfr.utils.io has no attribute 'io'
123. FAILED tests/unit/structural/test_safe_write.py::test_safe_write_binary_mode[False] - AttributeError: 'module' object at tnfr.utils.io has no attribute 'io'
124. FAILED tests/unit/structural/test_sense.py::test_sigma_from_iterable_vectorized_complex - assert 0 == 1
125. FAILED tests/unit/structural/test_sense.py::test_sigma_from_iterable_vectorized_empty - assert 0 == 1
126. FAILED tests/unit/structural/test_structural.py::test_run_sequence_triggers_dnfr_hook_every_operator - ValueError: Invalid sequence: missing coupling, dissonance, resonance segment
127. FAILED tests/unit/structural/test_structural.py::test_invalid_sequence - AssertionError: assert 'missing' in 'must start with emission, recursivity'
128. ERROR tests/unit/structural/test_io_optional_imports.py::test_io_optional_imports_are_lazy_proxies - ImportError: module tnfr.utils.io not in sys.modules
129. ERROR tests/unit/structural/test_io_optional_imports.py::test_yaml_safe_load_proxy_uses_fallback - ImportError: module tnfr.utils.io not in sys.modules
130. ERROR tests/unit/structural/test_io_optional_imports.py::test_toml_loads_proxy_falls_back_to_tomli - ImportError: module tnfr.utils.io not in sys.modules
```
