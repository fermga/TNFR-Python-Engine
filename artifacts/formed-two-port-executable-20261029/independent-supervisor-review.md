# Independent supervisor and control-protocol review

Date: 2026-10-29. Reviewer: `/root/admission_foundation`.
Supervisor/tests author: `/root/admission_guard`; protocol/harness author: `/root`.

**Verdict: PASS for the source-level supervisor and prospective finite-control
review. No native control, pytest invocation, scientific factory, target sign
evaluation, numerical step or producing worker was executed by this reviewer.**
The native controls remain to be frozen and run. This review does not authorize
the reserved reference attempt or establish that an actual source enters the
capture domain.

I authored the separate reference adapter and its tests. This is therefore not
an independent review of that adapter. My independent scope here is the Windows
supervisor, its tests, the root control declaration/harness, and consistency of
the proposed resource/admission documentation. I also suggested the corrections
recorded below; the review is substantive and not blind.

## Exact reviewed files

Paths are relative to `C:/TNFR-Python-Engine`; hashes bind the actual final bytes
read after the authors reported that these sources were stable.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `src/tnfr/research/bounded_worker.py` | 27702 | `f3bcc1c2bf9c7b364c7f90fb3734c38461210552c91a50767931a9fd2c2cf5d7` |
| `tests/research/test_bounded_worker.py` | 12673 | `74a9e4b93bc1ccf97ac6aba28b29210a22c8fec6b55f49dc223b56afa2fc992e` |
| `artifacts/formed-two-port-executable-20261029/controls-protocol.md` | 3255 | `e32a2c8c802601a696b8a84da5eeeed53561e9d3271fa3ee5ddc3dde887702af` |
| `artifacts/formed-two-port-executable-20261029/freeze_controls.py` | 2105 | `c690e1214b1005fe3dd27b411a0e452f57c7cd68af1484121538290fab68a622` |
| `artifacts/formed-two-port-executable-20261029/run_controls.py` | 2673 | `d21bba0759734242d88ed4d39ed753d522ea4aedc1c173c2bd6efde3b6c706f7` |
| `artifacts/formed-two-port-executable-20261029/admission_controls.py` | 8724 | `d8299337f934374e25b68e978ed9fec692e2607abe27cdcd1bedd2769a6309bc` |

Supporting local documents read:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `src/tnfr/research/artifact_io.py` | 10924 | `298708039b86876b1f70de7f8e49db787300850395d77cedccd1cf02fad5f011` |
| `.venv313/pyvenv.cfg` | 231 | `40ebda3e22180a3a02ce9bc42400467d5a200eae0efa78e6be7192d0422588b0` |
| `artifacts/formed-two-port-executable-20261029/chapter-append.md` | 5752 | `257e86d1d36dec4399f6c77c6a24b511e61d208cd4c8cbcd874bc6e734eb0a5f` |
| `artifacts/formed-two-port-executable-20261029/integrate.py` | 4819 | `d512ef29cce20cec60b86d7ee8c22c4b559bbcecb153f1e5e76b84b25172ba01` |

## Win32 types, creation and runtime identity

The ctypes declarations preserve pointer-sized HANDLE, SIZE_T and ULONG_PTR
values, DWORD/BOOL fields and 64-bit time/counter fields. On the Win64 ABI the
declared layouts imply BasicLimits=64, IOCounters=48, ExtendedLimits=144,
Security=24, Startup=104, StartupEx=112 and Process=24 bytes. These are a source
layout check, not an executed measurement of ctypes sizes. The declaration
matches the [basic-limit structure](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_limit_information),
[extended-limit structure](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information),
[startup structure](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/ns-processthreadsapi-startupinfow)
and [process-information structure](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/ns-processthreadsapi-process_information).

The selected flags are ACTIVE_PROCESS, PROCESS_MEMORY, JOB_MEMORY and
KILL_ON_JOB_CLOSE, without either breakaway permission. Both commitment caps
are set and queried before creation, with ActiveProcessLimit=1. They limit
committed virtual memory; the code correctly avoids equating this with RSS or
inferring a memory violation from an arbitrary child exit code.

The extended startup list retains its backing buffers through CreateProcessW.
Its HANDLE_LIST contains only the inheritable pipe writer and NUL input; the
reader and Job handle are not inherited. JOB_LIST associates the child during
creation. CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT,
EXTENDED_STARTUPINFO_PRESENT and CREATE_NO_WINDOW are correctly combined;
membership is checked before ResumeThread. Unsupported calls or nesting fail
closed. This follows the documented [process attributes](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute).

Launching the actual CPython image while retaining the venv executable in
argv and `__PYVENV_LAUNCHER__` avoids a redirector consuming the single-process
allowance. The bootstrap verifies resolved prefix/executable before runpy.
The local pyvenv.cfg specifies CPython 3.13.6, matching the reviewed
[venvlauncher implementation](https://github.com/python/cpython/blob/v3.13.6/PC/venvlauncher.c)
and [getpath handling](https://github.com/python/cpython/blob/v3.13.6/Modules/getpath.py).
In that implementation the launcher override is handled independently of
the ordinary environment flag; `-E` does not defeat this convention.

The child-side inspector queries the immediate Job and checks the same exact
limits. Microsoft documents NULL-handle queries as using the immediate Job
under nesting: [QueryInformationJobObject](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-queryinformationjobobject).
This proves neither the supervisor's identity nor an external wall deadline;
the owner explicitly preserves those limitations.

## Deadlines, cleanup and retained evidence

The total monotonic interval starts at supervisor entry. The reserve is inside
that interval: ten seconds for the 3600-second policy, leaving at most 3590
seconds for the child, including imports, admission and writes. A separate
thread owns a duplicated Job handle and can terminate the child while the
main thread is delayed draining output. Cleanup waits use the remaining total
interval. Job termination is an OS request and filesystem/scheduler delays
are not a hard-real-time guarantee; exhaustion is returned as failure.

The final source retains the Process object in the caller even if launch-time
membership verification fails. The caller observes termination before closing
its handles, or records incomplete cleanup. This resolves the original false
`worker_stopped_observed=True`/cleanup claim on that exception path. The new
test checks that a PID was retained and shutdown observed without resuming the
script. [TerminateJobObject](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-terminatejobobject)
is used for all members; ordinary process completion is separately observed.

The log drains bounded chunks and retains no more than its declared cap. Each
parent JSON record is serialized and size-checked before exclusive creation,
using the same exact-tree/shared JSON policy as its writer. Log <=1 MiB plus
two records <=512 KiB each fits the caller's 2-MiB reserve. An attempt directory
and partial immutable output cannot silently be reused.

`execution.json` is deliberately a pre-write snapshot. The returned result
checks elapsed time again after writing; only that post-write return can finish
attempt admission. A success snapshot alone is insufficient. The corrected
overrun test advances an injected monotonic offset only after the child and
watchdog have stopped and the snapshot was written; it no longer relies on a
1.5-second startup window. Other short native deadline controls remain actual
OS scheduling checks and must fail visibly if enforcement is unavailable.

## Control scope and prospective documentation

The declaration separates detached field/Jet arithmetic, one retained-target
sign re-admission, synthetic endpoint arithmetic, fabricated certificates and
inert OS workers. The admission-control fixture writes synthetic source and
review bytes and never imports those files or treats its fake review text as
scientific approval. The reviewed schedule controls are not reference steps.
The reference adapter itself still requires the other independent review.

The freeze includes all installed TNFR Python source, the six new maintained
files, selected retained evidence, instructions and test configuration, plus
the current stage's declarations/reviews/control scripts. It preserves a source
archive and verifies its members. The external harness verifies frozen current
bytes, reserves an exclusive log and applies its separate 300-second test
timeout. This is a finite control harness, not the scientific one-worker Job.

Two root-harness corrections are incorporated: success now requires a present
JUnit report with positive test count and zero failures/errors/skips, and a
partial/malformed JUnit report records a parsing error while preserving the
timeout/exit/runtime record and XML receipt. A zero pytest exit alone is no
longer sufficient. No controls have been run as part of this review.

The proposed chapter and programme replacements retain the distinctions among
code admission, the complete reference prefix, endpoint/tail sufficiency and
actual entry. They correctly explain centered form `||P_D x||_D`, the 3590/3600
budget split, immutable failure evidence and post-write admission. Applying
their statements about completed controls is conditional on those controls
actually passing; integrate.py checks that condition before writing owners.
The producing wrapper/caller/protocol still need their own exact freeze and
final review. This review does not claim those future bytes already exist or
grant a new root search, retry, policy change, autonomous contact law or
physical interpretation.

No unresolved blocking defect was found within this reviewed scope. Native
control results, source association and the final producing integration remain
separate obligations, with failure retained rather than converted into a
negative mathematical claim about entry.
