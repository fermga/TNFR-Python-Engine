# Passive transport measurement protocol

**Protocol version:** draft 6; scheduling references reviewed 2026-09-27.
Source reviews and frozen experiments retain their original dates and scopes.
**Admission:** `not_admitted`.
**Physical protocol result:** `not_tested`; no admitted measurement contract.
**Role:** supporting P2 measurement annex. The primary programme studies
generative coherence patterns and their possible physical interpretation;
MOSTR intake and additional data searches are deferred.
The [execution plan](FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) owns resumption.
The plan's [three-part variable contract](FIVE_STAGE_EXECUTION_PLAN.md#variable-definition-identification-and-evolution)
separates mathematical definition, physical identification and prospective
evolution. This annex supplies measurement evidence; it does not redefine
the variables or turn calibration into independent physical identification.
**Historical exploratory record:** the 2026-09-18 Volts comparison concerns
the earlier fixed-reference pure-EPI model and its nominal measurement map.
Its affine control has lower error within that comparison; it is not an
evaluation of the revised generative research programme. See the
[model boundary](#volts-model-boundary). Physical admission remains open.

**Scheduling:** no acquisition task is active here. The
[2026-09-21 source review](#terrestrial-source-review-2026-09-21) supported the
[first TCLab approximation](TCLAB_EXPLORATORY_PROTOCOL.md), now complete with
physical admission still open. The sole plan determines any resumption;
there is no automatic Volts refit or broad data search.
**Available resources:** computer only, as confirmed by the user. The route
is public terrestrial data. Building/acquiring a laboratory setup is not an
active dependency; the apparatus below remains a reference design.
The equal-capacity P2 model and a separately named fixed-reference variant
are implemented through one shared rational log/exp owner. Finding a scalar
dataset does not admit its physical mapping. The fixed-reference variant
also supports the explicitly exploratory, single-curve computation below.

This is the measurement annex for the plan's
[supporting P1-P5 bridge](FIVE_STAGE_EXECUTION_PLAN.md#supporting-measurement-bridge).
Historical O/S programme labels do not schedule current work. It specifies one
finite, falsifiable laboratory mapping of the canonical EPI channel. A
successful prediction would support that mapping within its measured domain;
it would not establish the origin of physical reality or a unique explanation
relative to other models with the same observable equation.

## Primary-source inventory

The bounded review used four search queries and inspected primary record
pages and JSON metadata on 2026-09-17. No signal archive was downloaded,
no signal was scored, and no outcome was used to choose a model. Publication,
record-update and acquisition dates are different; an unknown acquisition
date remains unknown.

| Candidate and primary source | License, access and dates | Structure and channels | Clock, calibration and holdout gate | Status and reason |
| --- | --- | --- | --- | --- |
| [Experimental networks of nonlinear oscillators, DOI 10.5281/zenodo.3521009](https://zenodo.org/records/3521009) | CC-BY-4.0; open. Published 2019-10-28; record modified 2020-01-24. Acquisition dates not established. | 28 electronic Rössler oscillators, 20 wiring configurations, one recorded state variable per node; 30,000 samples, 101 coupling settings, three repetitions. | Inspected metadata does not establish timestamp spacing, sensor calibration, complete hidden state, or an independent capacity/clock bridge. Whole repetitions could be reserved only after these gates and file identities are resolved. | `not_admitted`. Useful topology inventory; active multivariable dynamics cannot be assumed to be passive EPI diffusion. Coupling-setting labels are not calibrated canonical weights. |
| [Switched capacitor filter transient responses for fault detection, DOI 10.5281/zenodo.11122666](https://zenodo.org/records/11122666) | CC-BY-4.0 declared; files restricted. Published 2024-05-06; created/modified 2024-05-08. Acquisition dates not established. | Measured and simulated responses of second-order filters, with underdamped and overdamped cases. Exact wiring/channel map unavailable in inspected metadata. | Nominal training responses and labeled nominal/fault test responses are declared. Sampling clock, switching input, sensor calibration and split independence require documentation. Simulations must stay distinct from measurements; fault labels cannot enter calibration. | `not_admitted`. Access and causal/model specification unresolved; a driven switched filter is not the proposed isolated P2 transport experiment. |

License and access statements were checked against the corresponding
[3521009 metadata endpoint](https://zenodo.org/api/records/3521009) and
[11122666 metadata endpoint](https://zenodo.org/api/records/11122666).
A declared license does not grant access to restricted files. Access was not
requested and no account action was taken.

The earlier local inspection of `3521009` retained metadata and only its
9,991-byte `Structure.zip`: twenty connected 28-node, 42-edge graphs with
the same labeled degree vector. It downloaded zero time-series files. Its
local evidence is `artifacts/research/zenodo_3521009_admission/inspection.json`;
that ignored artifact is not a portable public evidence bundle. The primary
record advertises about 20.6 GB of files; this protocol does not authorize
downloading that collection.

No suitable simple passive RC or thermal time-series source was verified
within this search budget. The previously cited
[EPFL-hosted RC diffusion article](https://bigwww.epfl.ch/publications/sierociuk1501.pdf)
could not be retrieved during this review. It is a pending apparatus lead,
not an admitted data source, license determination or checked calibration.

### Follow-up admission search

The second bounded review used eight targeted electrical/thermal queries and
six primary-page inspections or retrieval attempts. It downloaded no signals,
opened no reserved response and admitted no additional dataset. A failed page
retrieval is an access limitation, not a finding about an unseen dataset.

| Primary source | Verified access or size | Reason it does not supply this protocol |
| --- | --- | --- |
| [Supercapacitor discharge measurements](https://zenodo.org/records/19221698), [metadata](https://zenodo.org/api/records/19221698) | Open CC-BY-4.0; ZIP 78,858,190 bytes; published 2026-03-25 | Active constant-current electronic load, not isolated charge sharing between two measured storage nodes; full clock/calibration contract not established |
| [INRIM 1 F supercapacitor measurements](https://zenodo.org/records/18348268), [metadata](https://zenodo.org/api/records/18348268) | Open CC-BY-4.0; 21 files, 274,853,203 bytes; published 2026-01-23 | One device with electrochemical test protocols; no verified simultaneous passive pair. The 270,585,991-byte training file alone exceeds the proposed download cap |
| [PhysLab capacitor apparatus](https://physlab.org/experiment/data-analysis-with-capacitor/) | Public teaching page; open-data license and raw file sizes not verified | One capacitor/resistor arrangement, not two observed storage coordinates; raw independent runs and instrument uncertainty not identified |
| [USGS thermal equilibration record](https://doi.org/10.5066/P15LYZOT), [official catalog](https://catalog.data.gov/dataset/data-for-examining-thermal-equilibration-rates-of-brook-trout-implanted-with-temperature-r) | Catalog declares public access; DOI retrieval failed; file sizes/license not verified | Bath-driven biological temperatures; isolated two-body/common-capacity assumptions unmet in the catalog description |
| [Han Hu Lab dataset inventory](https://ned3.uark.edu/datasets/) | Search-index description advertises thermocouple measurements; direct retrieval returned HTTP 502 | Isolation, exact files, license, calibration and repeat identities remain unverified; not an admitted source |
| [Dryad thermal electrosurgery record](https://datadryad.org/dataset/doi:10.5061/dryad.34tmpg4rv) | Public download links; advertised archive 273.22 MB; published 2024-08-16; license not verified in inspected content | Driven tissue heating and additional dissipation; incompatible passive P2 scope and oversized pilot archive |

These exclusions concern the chosen experiment. They do not establish that
other terrestrial TNFR mappings are impossible. Expanding the pressure law
to use a convenient dataset would require a separately derived and admitted
model; no such expansion is made here.

### Multinode source review — 2026-09-18

A bounded follow-up used four targeted queries and four primary-page
inspections/retrieval attempts. The selection question was whether a source
could support a frozen structural prediction across preparations or a declared
observation reduction, rather than another isolated scalar curve. No signal
files were downloaded or response plots/tables intentionally inspected.
Search snippets exposed published qualitative summaries; those summaries
were not used to choose a model or evaluation window.

| New lead | Verified metadata | Admission gap |
| --- | --- | --- |
| [Adelaide RC-ladder/PicoScope project](https://projectswiki.eleceng.adelaide.edu.au/projects/index.php/Projects:2020s2-7511_SQL_Database_for_Experimental_Metadata) | University project proposes measuring several RC ladder circuits with a PicoScope | No downloadable measurements, completed results, explicit circuit/channel map, uncertainty specification or independent preparation identities in the inspected page |
| [ETH BMHT RC-Ladder System Identification](https://www.gitlab.ethz.ch/BMHT/publications/rcladder-id), [official inventory](https://www.gitlab.ethz.ch/explore/projects/starred?non_archived=true&page=14&sort=latest_activity_desc) | Inventory describes analytical identification code, AGPL-3, linked to DOI `10.1109/TCSII.2023.3340505` | Direct repository retrieval failed; measured files, passive intervals, data license and independent preparations remain unverified. A code license is not a measurement-data contract |
| [Kiel spatial soil-temperature archive](https://opendata.uni-kiel.de/receive/fdr_mods_00000330), [authors' data article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12926585/) | Thermocouple time series and fixed sensor positions under controlled steady/cyclic loading around an electrical heater | Driven spatial transport is outside the current isolated pure-EPI model. Sensor positions do not specify a sufficient finite nodal state. Archive retrieval failed; exact files, sizes, calibration and independent preparations unverified |

No new candidate is admitted by this review. A failed retrieval gives no
finding about unseen file contents. A future multinode protocol must retain
enough initial-state information for prediction; hiding an indispensable
sensor is not a valid test of emergent dynamics. This inventory starts no
additional experiment or pressure-law extension.

<a id="terrestrial-source-review-2026-09-21"></a>
### Renewed terrestrial source review — 2026-09-21

The user explicitly requested real-world examples after reviewing the RC
proposal. This pass distinguishes access to useful measurements from admission
of a physical TNFR model. No fully documented passive RC network was admitted.
Driven thermal and active electrical data are useful alternatives for specific
questions; they cannot silently inherit the isolated two-node contract below.
No model was fitted or scored, no signal archive was downloaded, and no
reserved temperature/voltage arrays were inspected. Two small TCLab example
CSVs were inspected only for schema and timestamps. Published qualitative
summaries were seen; they are prior exposure, not blind evaluation evidence.

| Source and access | Verified observations | Useful question and admission boundary |
| --- | --- | --- |
| [TCLab / Pyomo.DoE](https://github.com/dowlinglab/pyomo-doe/tree/d250c5b0625afb35007c075df1e0f7125e74017d/data), [authors' apparatus/data description](https://dowlinglab.github.io/pyomo-doe/notebooks/tclab-model/) | Two heater/sensor assemblies; recorded seconds, two temperatures in Celsius and two heater commands in percent. Step/sine measurements and twelve repeat-labelled files are published. CSVs are about 20–29 kB. Repository BSD-3-Clause license; no separate data license found. | **First admission target:** a reserved temperature response to a specified input, with partial-state/memory analysis. Heating, ambient losses and latent heater temperatures require an explicit driven model. Percent command is not independently measured watts. Per-run acquisition provenance, resets, calibration, ambient conditions and uncertainty remain open. |
| [NED3-008: six thermocouples in a conduction apparatus](https://doi.org/10.5061/dryad.mcvdnckck), [metadata](https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.mcvdnckck) | CC0; latest advertised archive 253,223,992 bytes. Nine graphite and four titanium samples, with/without interface material, three labelled runs; seconds and six Celsius columns. Includes CAD and uncertainty notebooks. | Spatial observation and boundary/contact effects. Intended for steady-state conductivity; transient applicability is unverified. No heater-input column in the stated schema. README's 1000 Hz claim differs from the named [NI-9210's 14 S/s aggregate specification](https://www.ni.com/en/shop/hardware/temperature/model-ni-9210). Resolve logging versus fresh sensor conversions before defining a dynamical clock. Six sensors do not automatically define six closed thermal nodes. |
| [28 electronic Rössler oscillators](https://zenodo.org/records/3521009), [primary apparatus article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6961064/) | CC BY 4.0; twenty known graphs, 101 coupling settings and three repeated scans. The article now supplies nominal 30 kS/s acquisition and third-order 1500 Hz Butterworth filtering. Smallest signal archive: `R2.7z`, 1,027,983,572 bytes; topology archive only 9,991 bytes. | Phase, synchronization and hidden-state tests. One of three circuit variables is observed; active terms remain in its equation. Instrument accuracy, channel skew and filter treatment remain unresolved. Reserve whole scans, not dependent files within one scan. The article's zero-coupling filename conflicts with its indexing rule; retain that ambiguity. Large signal downloads remain deferred. |
| [Nonlinear analog learning network, 2024 data](https://doi.org/10.5061/dryad.8w9ghx3vx), [metadata](https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.8w9ghx3vx) | CC0; advertised data archive 15,756,671 bytes. The README exposes 4×4 node states under free/clamped conditions, edge gate-capacitor voltages and learning intervals; separate oscilloscope data have seconds and input/output volts. | How node response depends on changing connections and supplied training. This is an active, externally tasked learning system, not autonomous TNFR formation. Measurement-step snapshots are not automatically continuous full-state trajectories. Gate-voltage-to-conductance mapping, full timing, calibration and run independence need admission. Archive retrieval was not verified. |

**Resolved prior lead.** ETH's RC-ladder identification repository can now be
read through its [public GitLab API](https://gitlab.ethz.ch/api/v4/projects/41455/repository/files/README.md/raw?ref=master).
The [author paper](https://bretthannigan.ch/assets/pdf/Hannigan2023.pdf), section
III, uses 1,900 synthetic models; the inspected repository supplies code rather
than measured trajectories. Close the previous access blocker as a negative
measured-data lead. It remains mathematical identification work, not a physical
RC dataset. Other circuit descriptions without published voltage measurements
do not supply the missing observations.

**Pinned TCLab intake and leakage boundary.** Metadata was checked at commit
`d250c5b0625afb35007c075df1e0f7125e74017d`. The example files each contain
901 timestamps spanning about 900 seconds, with approximately one-second
spacing; this is a nominal recorded cadence, not a clock-accuracy certificate.
Git blob identities expose two aliases:

- `tclab_step_test.csv` and `validation_experiment_env_2_step_50_run_1.csv`:
  `3af095db6f6d13f6b80d0ab0c8cc4a57ca3928de`.
- `tclab_sine_test_5min_period.csv` and
  `validation_experiment_env_2_sin_5_50_run_1.csv`:
  `59a32387a0fd15d120df06531fefd84a1afdb373`.

They are the same evidence under different names and must never cross a
calibration/evaluation split. The twelve `env_2` filenames group four input
conditions by `run_1`, `run_2`, `run_3`; names alone do not prove independent
preparations. Do not interpret period/amplitude labels without their owner
specification. Published example outcomes are exposed; retain them as
development evidence rather than claim they are unseen.

**Follow-up completed under exploratory scope.** The user authorized a first
approximation while physical prerequisites remained unresolved. The
[TCLab experiment owner](TCLAB_EXPLORATORY_PROTOCOL.md) retains its declared
nodal/input model, acquisition caveats, frozen split, results and code.
It uses a separate driven affine realization; the equal-capacity P2 scorer
was not relabeled as a general thermal predictor. No author-fitted parameters
or thermal law were promoted to a TNFR derivation. The remaining sensor/clock,
ambient, input and initialization gaps still block physical admission.
Selection of this intake was based on accessible inputs, outputs and
repeated-file structure, not observed TNFR prediction error. The
[execution plan](FIVE_STAGE_EXECUTION_PLAN.md#supporting-measurement-bridge)
owns scheduling; the other candidates remain alternatives, not parallel
experiments.

Public metadata snapshots, file sizes/identities and review provenance are
retained locally in `artifacts/research/terrestrial_source_review_2026_09_21/`.
These ignored artifacts aid resumption but do not replace the cited public
owners or constitute an independently timestamped prediction. P2 remains
`not_admitted`; the source review itself supplied no performance result.
The later exploratory scores belong only to the linked TCLab experiment.

**Candidate-specific follow-up, 2026-09-26.** The
[regional phase/amplitude contract](PHASE_AMPLITUDE_MEASUREMENT_PROTOCOL.md#5-existing-source-admission-decision)
rechecks these four sources against the derived directed six-node law. None
is admitted for that prediction. That annex owns the new observation and
resolution requirements; this inventory and the frozen TCLab model are not
silently converted into a directed-diffusion experiment.

<a id="directed-source-decision-2026-09-26"></a>
### Bounded directed-source decision — 2026-09-26

The [regional measurement contract](PHASE_AMPLITUDE_MEASUREMENT_PROTOCOL.md)
motivated one bounded review of **six new candidate studies**, with four
targeted queries each for electrical, thermal and material transport sources.
An article and its linked data/methods records count as one candidate. This
completes the planned search; it does not start a recurring discovery task.
Only primary descriptions, methods and file metadata were assessed. Published
summaries and figure captions were exposed, but no signal workbook/archive,
response array or plot image was opened, no response scored and no model fitted.
Selecting a lead on that basis is not externally blind validation.

| Primary study / public record | Observations and access established | Decision for the fixed directed six-node prediction |
| --- | --- | --- |
| [Spatiotemporal diffusion metamaterial, 2020](https://www.nature.com/articles/s41467-020-17550-5) | Electrical apparatus with 50 modulated capacitors and 49 switches, maintained boundary voltage and time-averaged observations. Article CC BY 4.0; data statement points to paper/SI, with no linked machine-readable trajectory archive established. | Not admitted: changing storage/support and homogenization differ from the fixed generator. Independent preparation, averaging and simultaneous-state evidence are unresolved. |
| [Effective anyonic Bloch-oscillation circuit, 2022](https://www.nature.com/articles/s41467-022-29895-0) | Measured apparatus is an LC network of 23x23 node pairs with harmonic excitation and repeated node captures. The RC realization is an additional design. Article CC BY 4.0; supporting data/code require author request. | Not admitted: additional state, forcing, sequential captures and Hilbert phase do not supply the required scalar observation/law; no verified public raw archive. |
| [Electronically driven thermal pulses, 2026](https://www.nature.com/articles/s41467-026-72201-5) | Ten temperature nodes with programmed thermoelectric coupling; source XLSX advertised. Article CC BY-NC-ND 4.0; separate data-reuse terms, workbook size and independent run identities unverified. | Not admitted: a different actively controlled chain. Its reported linear settings have a negative neighbor coefficient, outside the nonnegative pure-EPI realization; see the exact sign check below. |
| [Temporal anti-parity-time transport, published 2025 / issue 2026](https://www.nature.com/articles/s41567-025-03129-8) | Rotating thermal architecture with timed changes in convection/material control and advertised figure-data XLSX files. Independent complete-state acquisition schema, clock uncertainty and data license not established. | Not admitted: time-programmed distributed dynamics require their own observation/reduction law. Source-data availability alone does not establish a closed fixed six-node generator. |
| [MOSTR apparatus, 2026](https://doi.org/10.1016/j.ohx.2026.e00776), [versioned dataset](https://data.mendeley.com/datasets/z24ggxmg5f/3) | Three stirred tanks in series with outlet tracer measurements. Public CC BY 4.0 workbooks include a triple-system experiment and separate conductivity/flow calibrations. | Not the triangle model. **Selected only for a bounded alternative-graph admission check**, owing to known one-way connectivity, accessible measurements and calibration files. Physical admission remains open. |
| [Stirred-tank particle tracking, 2026 preprint](https://arxiv.org/html/2607.26813v1), [experimental-data DOI](https://doi.org/10.15480/882.17469) | Particle trajectories in a laboratory vessel; compartment networks require an additional inference. Data resolver returned 404 in this review. [Analysis-code record](https://zenodo.org/records/21874120) does not establish experimental-data access or its license. | Not admitted: particle positions are not independent scalar compartment-state observations. Deriving the graph from the evaluated trajectories would need a separate calibration/reduction and holdout contract. |

**Constitutive sign check, not a new pressure implementation.** The thermal
pulse study declares a linear row, in physical seconds,

\[
\dot T_i=(C_d+C_v)(T_{i-1}-T_i)+(C_d-C_v)(T_{i+1}-T_i),\qquad C_d>0.
\]

Applying the existing [maximum-principle criterion](../nodal/PRESSURE_CONSTITUTIVE_SCOPE.md#4-what-locality-and-symmetry-can-derive)
to these observed nodes in a common positive affine temperature chart gives
nonnegative pure-EPI weights **if and only if** `abs(C_v)<=C_d`. Their sum is
`2*C_d`, so with structural time `t=s/tau_*`, capacity would be
`nu=2*tau_*C_d`. The reported linear pairs `(0.006,0.010)` and
`(0.003,0.053)` instead give negative right-neighbor coefficients `-0.004`
and `-0.050` per second. Positive capacity rescaling cannot remove that sign;
clipping would change the apparatus law. This rules out that specific mapping,
not physical feasibility, all other TNFR pressure laws or arbitrary hidden-state
reductions. No signed-conductance exception is added to the engine.

**Decision:** park the exact six-node physical test at this source boundary.
Its mathematical result, observation contract and uncertainty criterion remain
valid under their hypotheses. No evidence here falsifies every TNFR model or
proves that a suitable public dataset does not exist. The selected alternative
below tests prospective nodal transport, not the triangle phase-locking theorem.

#### MOSTR: selected intake, not physical admission

The [dataset](https://data.mendeley.com/datasets/z24ggxmg5f/3), DOI
`10.17632/z24ggxmg5f.3`, was published on 2026-02-19. Its
[experiment metadata](https://data.mendeley.com/public-api/datasets/z24ggxmg5f/files?folder_id=535e9331-6894-489d-9949-4b35faa9e0cc&version=3&%24start=0&%24limit=1000)
and [calibration metadata](https://data.mendeley.com/public-api/datasets/z24ggxmg5f/files?folder_id=aaf814db-dd55-4703-8c2f-7e2ac58dab86&version=3&%24start=0&%24limit=1000)
advertise these files; their bytes have **not** been downloaded or checked:

| File | Advertised bytes | Version-3 file identity |
| --- | ---: | --- |
| `MOSTR_Triple_System_Residence_Data.xlsx` | 428,951 | `e399c85d-e0a5-4234-9ab2-f8c374e55637` |
| `MOSTR_Conductivity_Probe_Calibration.xlsx` | 50,786 | `672a6653-dc05-4c67-9572-0dd4d2ba29ba` |
| `MOSTR_Flowrate_Calibration_Example.xlsx` | 91,764 | `a7ecc20e-b92c-4983-9dae-3f0e043cce7b` |

The first file's advertised SHA-256 is
`a3bfc67847879452f5f6602397befd2ff954e6b2e52b4d24af3e95381082f860`.
This is repository metadata, not local byte verification or a forecast seal.

The [author manuscript](https://eprints.whiterose.ac.uk/id/eprint/241093/)
describes the triple experiment with balanced flow `0.175 L/min`, hydraulic
head `20 cm` and tracer injected at the first vessel's inlet. Its generic
procedure separately describes a `20 mL` injection through a vessel lid; those
descriptions do not establish the same event geometry or concentration waveform.
The paper describes five-second IoT averages; the triple workbook's actual
windows remain unverified. Example/representative calibrations are not
automatically the actual three sensors' calibrations. The teaching-session
record lacks hydraulic heads; it is not substituted for the triple workbook.

**Conditional model, an inference to check rather than an apparatus finding.**
For fixed positive effective volumes `V_i`, balanced through-flow `Q`, ideal
mixing and independently specified inlet concentration `c_0(s)`, conservation
of tracer would give

\[
V_i\frac{dc_i}{ds}=Q(c_{i-1}-c_i),\qquad
x_i=(c_i-c_{\rm ref})/c_*,\qquad
p_i=x_{i-1}-x_i,\quad \nu_i=\tau_*Q/V_i.
\]

Here `c_*>0` and `t=s/tau_*`. The existing directed pure-EPI row reads one
upstream neighbor, opposite the material-flow arrow under engine conventions.
This is a proposed physical bridge to an existing model, not a derivation of
mixing physics from the nodal product. Its inlet and drain are supplied
boundaries. An unresolved injection is an unresolved input event; it must not
be inferred from the reserved response or silently replaced by an exact impulse
or desired initial condition. A post-injection forecast instead needs an
independently justified cutoff, initial state and later upstream baseline.

For constant inlet, deviations have a triangular generator with real diagonal
eigenvalues `-Q/V_i` in physical time `s`, or `-tau_*Q/V_i` in structural time.
Repeated rates permit polynomial-times-exponential terms.
The [directed dynamics owner](../TNFR_DIRECTED_NONNORMAL_DYNAMICS.md) supplies
the relevant nonnormal scope. The two-triangle complex modes, phase lag and
frozen-ratio discriminator do not transfer to this cascade. Its question is
whether independently specified local transport predicts downstream response
with the same observation model and no response-dependent pressure.

Before prediction, the schema must establish sensor-to-vessel identities and
calibration applicability, conductivity-to-concentration/temperature mapping,
effective volumes, balanced flow, injection timing, initial-state information,
averaging windows/timestamp alignment and independent preparation identities.
Bypass, stagnant regions, sensor lag and incomplete mixing are possible closure
failures, not fitted corrections to add after evaluating a response. A first
five-second average is not automatically the instantaneous initial state; a
forecast must apply the declared averaging observation as well. If these
dependencies or a separate evaluation acquisition are absent, retain the
precise gap and do not claim physical P3 admission. The
[execution plan](FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) alone schedules
that bounded schema check, now deferred under the collective-emergence priority.
The frozen TCLab comparison remains unchanged.

<a id="computer-only-candidates-and-model-selection"></a>
## Historical scalar-source selection — 2026-09-18

The decisions below belong to the completed Volts exploration. They are
preserved for provenance, not instructions to reopen that intake or override
the current source decision.

The focused scalar-data search retained two actionable leads. A single
recorded curve is not two independent preparations.

| Candidate | Pinned observation metadata | Remaining admission gaps |
| --- | --- | --- |
| [Stat2Data / Volts manual](https://raw.githubusercontent.com/statmanrobin/Stat2Data/3fe987c7e1813fd9960a3ca9293873359c1e07e9/man/Volts.Rd), [package/license](https://raw.githubusercontent.com/statmanrobin/Stat2Data/3fe987c7e1813fd9960a3ca9293873359c1e07e9/DESCRIPTION) | Package 2.0.0, GPL-3; 50 author-measured voltage/time observations, nominal 0.02-second spacing; one documented capacitor discharge after battery charging. Author-repository commit `3fe987c7e1813fd9960a3ca9293873359c1e07e9`; `data/Volts.rda`, 767 bytes, Git blob SHA-1 `6602a54be75afdec50b3a179457fc2af483bf29c` | No documented discharge wiring/source isolation, fixed-reference verification, voltmeter/clock accuracy, or independent run identities. Accessible exploratory candidate, not a reserved physical-validation dataset |
| [Low-Energy-Meter](https://github.com/duerrfk/low-energy-meter), [directory metadata](https://api.github.com/repos/duerrfk/low-energy-meter/contents/data) | `no_load-f1Hz.csv`, 614,400 bytes; Git blob SHA-1 `c2ea34ca95407bdccb7e82f55b192f68094d4adb`. README describes nanosecond timestamps, discharge-cycle IDs and ADC counts; the no-load file concerns leakage/instrument loading | No explicit data license or verified independent repeat count, sensor/clock bounds or constant-reference evidence. README contains contradictory inverse versus proportional count-to-voltage formulas. Published README outcome summaries have been seen; this is not an untouched blind source |

The Volts file is fixed at
[its exact revision](https://raw.githubusercontent.com/statmanrobin/Stat2Data/3fe987c7e1813fd9960a3ca9293873359c1e07e9/data/Volts.rda).
The historical admission preparation selected this 767-byte file for caching, with
a 1,024-byte transfer cap and a Git-blob identity check before retention.
The initial cache was not parsed, fitted or scored. That byte/provenance record
is `artifacts/research/stat2data_volts_admission/inspection.json`;
it records the actual raw-data SHA-256 without admitting a physical mapping.
The retained 767 bytes have SHA-256
`e50e505783dbf9b6d477165c92328e805e5d8bb0d27e3458862e2d5ba7f00a52`.
The initial cache delivery performed no conversion/decoding. Draft 4 later
permitted the bounded exploratory decoding specified below. Nominal sample
spacing is not clock accuracy.

The pinned package `DESCRIPTION` and the
[official CRAN index](https://stat.ethz.ch/CRAN/web/packages/Stat2Data/index.html)
declare GPL-3. The generated
[package overview](https://search.r-project.org/CRAN/refmans/Stat2Data/html/Stat2Data-package.html)
still says GPL-2; retain the pinned release declaration rather than silently
mixing these metadata versions. The follow-up primary-source review did not
resolve the apparatus, uncertainty or independent-preparation gaps.

Do not repair missing independent preparations by splitting this one curve
and calling its windows independent runs. It can support a separately labeled
exploratory study only after its narrower observation/model scope is declared.
Physical P3 admission retains its missing calibration and independence gates.

### Predeclared Volts exploratory delivery (before decoding values)

This is a single fixed computation inside P2.3, not physical P3 admission.
The mathematical fixed-reference implementation is reusable; the following
observation map remains a conditional hypothesis for this incompletely
documented source. Stop after this one specification, including an unfavorable
or unavailable result. No reference, split, horizon or error allowance is
selected after looking at the curve.

- Verify the pinned SHA-256 and 767-byte file before decoding. Bound expanded
  serialization to 65,536 bytes. Use a data-only R serialization decoder, not
  R evaluation. Require exactly one `Volts` data frame, 50 finite numeric rows,
  `Voltage` and `Time` columns, and increasing times. No downloads, gap
  interpolation, smoothing, row deletion or synthetic replacement in the run.
- Read voltages as nominal scalar EPI coordinates (unit scale, zero offset)
  and reported seconds as nominal structural time (bridge one). These are
  stated chart/clock conventions, not sensor or physical-clock calibration.
  Set the candidate constant reference to zero before decoding; this is an
  unverified fixed-reference hypothesis, never a fitted asymptote.
- Fit only the first and 25th samples (zero-based indices 0 and 24) using
  the continuous fixed-reference logarithmic identity. The first 25 rows
  are calibration; rows 25 through 49 are the continuation comparison.
  Interior calibration values may be used only by the declared comparison
  baseline, not to select another nodal model or rate after evaluation.
  Unresolved sign/decay or invalid times ends the specified nodal analysis.
- Freeze the nominal rate enclosure and save a content-hashed continuation
  from the 25th value at the remaining reported timestamps before scoring
  continuation voltages. Reading that time schedule is declared metadata
  access, not evidence that these public samples were acquired prospectively.
  Predict through the existing rational exponential owner; no nodal Euler
  trajectory is invented or fitted to the continuation.
- Compare coordinate RMSE/maximum absolute error and unchanged-initial-value
  persistence; include a calibration-only affine AR-1 baseline if its fit is
  identifiable, otherwise report baseline unavailability. These are descriptive
  controls, not additions to TNFR pressure. Save every specified result.
- Instrument error, clock error, physical-reference adequacy and a physical
  acceptance threshold remain **unknown**, not zero. A rational enclosure of
  nominal arithmetic is not a measurement confidence interval. No numerical
  error, favorable baseline comparison or within-curve continuation closes
  the independent-preparation or physical-admission gates.

The saved protocol hash and result must retain `within_single_acquisition`,
`physical_status=not_admitted` and `measurement_verdict=not_assessed`.
The original byte-only inspection remains unchanged as historical evidence.

<a id="volts-exploratory-record"></a>
### Completed exploratory record — 2026-09-18

<a id="volts-model-boundary"></a>
**Historical model boundary.** This experiment predates the current foundation
reassessment. It evaluated `x'=nu*(0-x)` with fixed positive `nu`, a zero-capacity
reference node, pure EPI pressure and no active phase, capacity-contrast or
topology source. Voltage was assigned directly to scalar EPI and reported
seconds to structural time; neither mapping was independently calibrated.
Those premises identify the tested model more precisely than a release label.
The fixed-reference derivation remains a valid conditional reference; the
experiment did not evaluate the complete engine or the current search for
justified generative closure and identity formation.

Retain its data, frozen specification, source provenance and unfavorable
comparison as historical evidence about that model/map pair. Do not transfer
the outcome to a revised candidate without establishing equivalence of the
tested observable prediction and measurement assumptions. Conversely, changing
the programme or release name does not invalidate an unchanged prediction.
A new candidate needs its own declared model boundary and independent reserved
evaluation; these already inspected values cannot become fresh holdout data.
Retiring the earlier candidate as the programme's representative does not
provide the physical evidence still missing for the revised one.

The [bounded reader](../../benchmarks/volts_data.py) and
[fixed-design benchmark](../../benchmarks/volts_fixed_reference_exploration.py)
executed the specification above without scientific changes. The archive
expands to 1,016 bytes, R serialization version 2. Only one nominal rate was
estimated: `nu` lies in the arithmetic interval
`[2.1590717641670265, 2.159071764167027]` per reported second. This narrow
interval encloses numerical arithmetic, not measurement uncertainty.
The forecast starts at row 24 (`t=0.48 s`, `V=3.2682 V`); scoring uses only
the 25 later observations through `t=0.98 s`.

| Frozen predictor | Continuation RMSE (V) | Maximum absolute error (V) |
| --- | ---: | ---: |
| Pure-EPI fixed-reference P2 | 0.0769495843 | 0.129818675 |
| Persistence at the initial value | 1.3676904144 | 2.028000000 |
| Prefix-only affine AR-1 control | 0.0255576220 | 0.046371615 |

The historical fixed-reference continuation improves on persistence but is
less accurate than the declared affine control. It shows no predictive
advantage for that specified model/map pair on this trace. It neither
identifies the cause of the residual nor evaluates revised TNFR realizations:
the observation map, constant reference,
apparatus and error bounds remain unverified. No offset, rate, reference,
split or pressure law was changed after scoring. Do not mine this curve for
a replacement model and call the same continuation newly reserved.

The decoder initially stopped on historical unmarked text encoding, before
issuing or scoring a forecast. A synthetic-tested strict ASCII metadata
fallback repaired ingestion; it changed no measurements or scientific
settings. Preserve this failed attempt alongside the successful execution.
The whole archive was decoded before issue; the predictor received only
prefix values and the reported future time schedule. This is explicitly
`within_single_acquisition`, not blind prospective evidence. All physical
statuses remain `not_admitted` / `not_assessed`.

Local retained bundle: `artifacts/research/volts_exploration_2026_09_18/`.
Its execution manifest records base commit
`ce5d40feead702fc1a77fbea128ffa836a859081` with uncommitted source identified by
`sha256:f63bee1f1eeaed9e456224cf66530d483f8eebec2421f5ae3e093d8baddd9efa`.
The commit alone does not identify the executed source. Its predictor used the
rational exponential reference, without a live graph trajectory.
The exact pre-value protocol/specification snapshots are immutable:

- `protocol_before_values.md`, SHA-256
  `e2b584003c2e5605ae786a533a48cb067a002e67f00a5ba9de2be8fc1f1c0124`.
- `spec_before_values.json`, SHA-256
  `b6a56c8856ea339778c6d6dfffd4f256a03f1fa2ce2a5e398aefb457afe41da0`.
- `attempt_01.json`: decoder stop; no forecast or score.
- `run_v2/`: raw data, copied protocol/specification, ingestion metadata,
  issued forecast, complete residuals and byte-checked evidence envelope.
  Forecast content hash:
  `sha256:57bb1e059470d80dbd6e685c6b5458857f702390018ccb31e080656f6102990c`.

These are local ignored artifacts, not a published bundle or an independently
authenticated timestamp. Python >=3.11 and `pip install -e ".[research-data]"`
provide the pinned decoder. A replay uses the retained inputs and a fresh
output directory; it cannot become a new blind study:

```powershell
python -X utf8 benchmarks/volts_fixed_reference_exploration.py `
  --data artifacts/research/stat2data_volts_admission/Volts.rda `
  --spec artifacts/research/volts_exploration_2026_09_18/spec_before_values.json `
  --protocol artifacts/research/volts_exploration_2026_09_18/protocol_before_values.md `
  --expected-spec-hash b6a56c8856ea339778c6d6dfffd4f256a03f1fa2ce2a5e398aefb457afe41da0 `
  --output artifacts/research/volts_exploration_2026_09_18/replay
```

The combined focused suite passes 174 tests: both mathematical models,
the unchanged equal-capacity acquisition contract, bounded ingestion,
prefix isolation, baseline failures and complete evidence recording.
The source-bound local checkpoint is
`artifacts/research/fixed_reference_volts_validation_2026_09_18.json`.
Tests validate these software contracts, not the physical hypotheses.

### Derived fixed-reference candidate

On the same fixed symmetric positive P2 support, select **pure EPI pressure**
and capacities `(nu,0)`, with `nu>0`. All other pressure channels are inactive;
otherwise the capacity difference itself could contribute another channel.
The canonical nodal equation gives

```text
p = (r-x, x-r),           x' = nu*(r-x),           r' = 0
d = x-r,                 d' = -nu*d
x(t) = exp(-nu*t)*x(0) + (1-exp(-nu*t))*r.
```

This follows from the nodal law without importing the circuit or cooling
equation. The arithmetic mean is NOT conserved; the reference `r` is.
An inactive node may retain nonzero pressure: stationarity is not structural
equilibrium. The existing diffusion energy owner accepts degenerate mobility;
positive-capacity certificates using `d_i/nu_i` do not cover this boundary.

Physical use requires an independently specified constant reference and
support, adequate state observation and bounded external loading. Flat bath
measurements do not prove that a bath is a zero-capacity TNFR node; an active
controller may instead be an external input. A varying reference requires a
distinct driven model and its complete input history, not a fitted final
asymptote. No held-out value may estimate `r` or the clock conversion.

For this candidate, existing log/exp enclosures can be reused with contrast
`x-r` and decay factor `exp(-nu*t)`. The capacity ratio divides by elapsed
time, not twice elapsed time. There is no constant-mean rejection rule.
Do not fabricate a second observed channel or reuse the equal-capacity
adapter unchanged. The mathematical implementation now lives in
[p2_transport_reference.py](../../src/tnfr/physics/p2_transport_reference.py):
`bound_fixed_reference_capacity` and `bound_fixed_reference_transport` reuse
the same contrast/log, decay/exp and convex interval kernels as the equal-
capacity model. The reference may be a declared interval and the initial
contrast may have either strict sign. The returned outer intervals do not
prove one common latent fit. These functions do not mutate a live graph or
admit empirical observations. Physical admission still requires the source
contract; this remains one candidate within P2.3, not another queue.

## Reference apparatus and canonical equal-capacity prediction

The reference design uses two nominally equivalent capacitive elements, observed
simultaneously, prepared separately and then interconnected through one
fixed passive link. Charging sources are disconnected before the declared
observation interval. Record wiring, switch timing, sensor input loading,
component characterization and ambient conditions. Equivalence and negligible
external forcing over the finite interval require measured bounds; component
labels alone do not establish them. The initial experiment changes only the
prepared contrast, keeping the link and apparatus fixed.

The hardware is a candidate realization, not an imported circuit law.
The proposed TNFR model is the isolated EPI channel on fixed P2 support,
with one common positive capacity and no active phase or capacity-feedback
channel. Each physical voltage is mapped to a signed EPI coordinate `x_i`
by a fixed sensor scale and offset. Both channels must share the same
structural unit after independently calibrated sensor correction. Define
structural time by `t = b * (tau - tau_0)`, where `tau` is the independently
checked acquisition clock and `b > 0` is fixed before capacity estimation.
A declared unit convention for `b` is permitted if labeled as a convention;
it is not evidence of a universal physical-to-structural conversion.

The [canonical diffusion owner](../../src/tnfr/physics/structural_diffusion.py)
then gives, without using a circuit governing law as an axiom,

```text
p_1 = x_2 - x_1,                 p_2 = x_1 - x_2
dx_1/dt = nu * p_1,              dx_2/dt = nu * p_2,       nu > 0
m = (x_1 + x_2)/2,               d = x_1 - x_2
dm/dt = 0,                      dd/dt = -2 * nu * d
d(t) = d(0) * exp(-2 * nu * t)
x_1(t) = m(0) + d(t)/2,          x_2(t) = m(0) - d(t)/2
```

These are predictions of the stated nodal model. Pressure is calculated
from the two observed coordinates, never defined as a measured derivative
divided by fitted capacity. Only the product `nu * b` is identifiable from
a physical-time contrast trace if both are unknown. Fixing the clock bridge
is therefore an admission condition, not a second fit parameter. Estimate
one positive `nu` from calibration runs alone and freeze its uncertainty
set. Nonpositive or unidentifiable capacity blocks this positive-capacity
model; it is not clipped into its domain.

For P2, uniformly scaling the single positive edge weight leaves `L_rw`
unchanged. Changing a resistor or link strength cannot be represented merely
by rescaling that edge at fixed capacity. This first protocol consequently
does not sweep link strength or infer a universal relation between capacity
and hardware component values. An intervention is a recorded preparation,
not automatically a canonical glyph or a U3 certificate.

## State, clock and measurement admission

| Item | Required frozen specification | Current missing evidence |
| --- | --- | --- |
| State map | Two sensor identities; common EPI unit; fixed offsets/scales; range and linearity bounds | Apparatus and sensor calibration absent |
| Time | Raw timestamps, synchronization/skew bounds, finite positive bridge, origin rule and gap policy | Physical clock, sampling period and uncertainty absent |
| Support and input | Known P2 wiring, fixed link, source disconnection, observation interval and loading/leakage bounds | Wiring and external-input characterization absent |
| Capacity | Calibration-only common positive rate and identified uncertainty; no outcome-dependent choice | No calibration runs |
| Phase | Unavailable as an oscillatory observable; phase channel explicitly inactive in the selected model | Physical adequacy of that restriction untested |
| State sufficiency | Actual generator and observation map; distinguish unknown source terms from measurement noise | Model-specific apparatus check pending |
| Reserved observations | Disjoint preparation/run IDs, hashes, initial-sample allowance and fixed forecast horizon | No acquired runs or reserved data |
| Decision | Predeclared joint finite-trajectory acceptance tube, rejection and abstention rules | Instrument, calibration and numerical error budgets unset |

With both EPI coordinates measured, the selected two-state model has
observation matrix `C = I`. Its algebraic state observability is immediate;
this does not prove that the apparatus has only those two relevant states.
With contrast alone, the mean is hidden but the contrast equation closes
under the common-capacity hypotheses. Do not promote that to a full-state
claim. Reuse [observability](../../src/tnfr/physics/observability.py) on the
actual generator and observation map, and the hidden-source distinction in
[derived EPI memory](../../src/tnfr/physics/epi_memory.py) if a reduction is
proposed. An algebraic rank result supplies neither noise conditioning nor
an instrument error bound. A material unmodeled leakage/loading contribution
requires a revised admitted model, not a retrospectively invented pressure.

## Calibration, forecast and finite decision

Use separate whole preparations for calibration and evaluation. Calibration
may estimate only the declared common capacity after sensor and clock
mapping are fixed. Freeze the fit rule, accepted domain, parameter interval,
data identities and code revision. No graph, normalization, filter, channel,
capacity, tolerance or horizon is selected using a reserved outcome.

The evaluation initialization allowance is the first synchronized sensor
pair, with its uncertainty, at a predeclared forecast origin after the
switching interval. Choose this origin from the hardware timing/calibration
rule, not from the later relaxation trace. Forecast all subsequent requested
timestamps before reading their values. No full-record centering, Hilbert
transform, interpolation across unexplained gaps or future-aware smoothing
is needed for this prediction.

Primary observation: the held-out contrast trajectory under the frozen
capacity and clock bridge. The constant-mean prediction is a separately
reported check of the same admitted model, not an ignored nuisance when it
fails. Construct a simultaneous finite-trajectory tube from the declared
initial-state, sensor, time, capacity and numerical uncertainty. A collection
of pointwise confidence intervals is not automatically such a tube. If the
uncertainty cannot be bounded at the declared scope, report `inconclusive`.
Exceeding the frozen acceptance rule rejects this mapping on these runs;
agreement supports only its declared finite domain. Report every reserved
run, including failures, missing samples and exclusions with reasons.

## Implemented continuous-time uncertainty boundary

The numerical part of admission now has an executable owner:
[p2_transport_reference.py](../../src/tnfr/physics/p2_transport_reference.py),
with measurement/run integration in
[validation/p2_transport.py](../../src/tnfr/validation/p2_transport.py).
It uses the existing rational logarithm and negative-exponential enclosures;
it introduces no pressure term, circuit governing law or new nodal solver.
All uncertainty bounds below are supplied hypotheses. Their instrumental
justification is still missing and remains a physical admission gate.

### Continuous capacity, distinct from the Euler control

The earlier P1 finite-increment fit remains an engineering control. Even for
an ideal noiseless P2 trajectory sampled at a common structural step `h`,
its fitted coefficient is

```text
nu_Euler = (1 - exp(-2*nu*h)) / (2*h) < nu,     nu>0, h>0.
```

It cannot be silently relabeled as the continuous physical rate. Irregular
pooled increments do not in general have this single-step conversion.
The new calibration uses each declared run's fixed first and last samples;
interior samples are not fitted, and endpoint consistency does not validate
the whole calibration trajectory. Durations and endpoint selection must not
be changed after opening reserved outcomes.

After fixed sensor conversion, suppose the initial and final contrast have
one resolved common sign. Reverse both signs if negative, giving positive
intervals `D0=[a,b]` and `D1=[c,d]`. For elapsed structural time
`T=[sL,sU]`, `sL>0`, require resolved decay `d/a<1`. Then

```text
r = D1 / D0 is enclosed by [c/b, d/a] = [rL,rU]
nu is enclosed by [-log(rU)/(2*sU), -log(rL)/(2*sL)].
```

Rational log bounds round the interval outward. Initial/final mean intervals
must overlap. Each run must resolve nonzero contrast and positive decay;
incompatible, sign-ambiguous or unidentifiable inputs raise explicit errors.
Intersect all run capacity enclosures. A nonempty intersection is a necessary
outer consistency condition, not proof of one joint latent fit. Finally round
the capacity bounds outward to binary64 rational endpoints to limit arithmetic
growth; a lost positive lower bound or infinite upper bound blocks this path.
This computational enclosure is separate from measurement uncertainty.

### Reserved joint outer tube

`P2MeasurementBounds` freezes two offsets/scales, total EPI-coordinate errors,
timestamp error and a positive structural-time bridge interval. Coordinate
errors include uncertainty in the sensor conversion itself. A bound `eps_t`
on each timestamp gives elapsed-time error `2*eps_t`; time zero is the same
initial event, so its duration interval is exactly zero. Unresolved positive
durations are rejected. The declared sensor/clock bounds must hold jointly
throughout the full finite experiment; no confidence level is inferred.
Both channels in each pair must represent the same latent observation time.
Acquisition skew needs synchronization evidence or an independently derived
coordinate-error allowance for time alignment. Timestamp error alone cannot
align asynchronous channels or justify their constant-mean test.

For capacity in `[nuL,nuU]` and elapsed time in `[tL,tU]`, the decay lies in
`[exp(-2*nuU*tU), exp(-2*nuL*tL)]`, evaluated with rational enclosures.
Coordinates use the positive semigroup directly:

```text
x1(t) = ((1+e)*x1(0) + (1-e)*x2(0))/2
x2(t) = ((1-e)*x1(0) + (1+e)*x2(0))/2,   e=exp(-2*nu*t).
```

Monotonicity in the initial coordinates and corner extrema in `e` give
sound boxes while preserving the mean/contrast dependency. The mean and
contrast are also retained as separate observable tests. The shared exponent
cap is 4096 and the reference accepts at most 10,000 samples; these are
computation guards, not physical constants. Rational output endpoints must
not be converted to ordinary floats and called certified bounds.

`calibrate_p2_transport` checks the known undirected two-node positive edge;
`forecast_p2_transport` consumes only frozen calibration, reserved identities,
initial measurements and timestamps. The latter must be a predeclared schedule
or clock metadata available without inspecting future response values.
`write_p2_transport_forecast` saves exact rational endpoints as hexadecimal
numerator/denominator strings; retain its content hash before opening outcomes.
`score_p2_transport` reuses the whole-acquisition separation and requires that
issued hash, matching units/times and unchanged initialization.

Disjoint coordinate, mean or contrast boxes give
`incompatible_with_declared_bounds`. If every outer box overlaps, the result
is only `not_falsified_by_enclosures`: different boxes may admit different
latent parameters. That is neither a joint-feasibility proof nor statistical
acceptance. The physical status remains `not_admitted_by_this_score` in both
cases. Excessive tube width requires a preregistered resolution/power decision;
it cannot become positive evidence just because all observations overlap.

Independent analytic/Decimal fixtures and wrong-mean, wrong-identity and
changed-forecast controls exercise this boundary. These are software/model
tests, not measured data. Existing manifest/sidecar owners can bind the saved
bytes without acquiring a second evidence ledger. P1's Euler tests and their
historical checkpoint remain unchanged.

## Proposed pilot resources, not instrument tolerances

The following are provisional engineering caps for admission planning.
They do not authorize acquisition with unresolved physical tolerances and
are not TNFR constants or a statistical power calculation.
Under the computer-only constraint these apparatus/preparation rows are
requirements a matching public archive would need to document, not tasks
for the user to build or acquire equipment.

| Resource | Proposed cap and stop rule |
| --- | --- |
| Apparatus | One P2 setup, two synchronized observation channels, one unchanged link |
| Preparations | Two calibration runs at one nonzero prepared contrast; two held-out contrast conditions with three independent preparations each: eight total |
| Record size | At most 10,000 paired samples per run and 100 MiB for the complete raw/calibration/prediction bundle |
| Acquisition duration | At most 30 minutes per run; choose a shorter fixed horizon only from apparatus/calibration evidence before held-out acquisition |
| Data search/download | This metadata review is complete. Before any later signal download, pin exact files and compressed/expanded bounds; proposed pilot caps 100 MiB compressed and 250 MiB expanded |
| Computation | One frozen calibration specification and one prediction per reserved preparation; no automatic refitting or larger campaign after rejection/inconclusiveness |

Sampling interval, voltage/temperature limits, switch settling allowance,
clock accuracy, sensor skew, error bars and acceptance thresholds remain
unset. They must be supplied by the selected instrument and calibration,
not by these sample/storage budgets. If an adequate record cannot fit the
caps, revise and version the protocol before observing reserved outcomes.

## Evidence bundle and current exit gate

Reuse [CoreExperimentManifest](../../src/tnfr/research/core_manifests.py)
and [EvidenceSidecar](../../src/tnfr/research/evidence_sidecar.py). Retain
this annex, exact raw/calibration/evaluation file identities, source revision,
raw timestamps/masks, immutable fit, forecasts and complete result ledger in
one accessible bundle. `validate_metadata()` checks declarations only;
`validate_for_admission(root_dir=...)` additionally checks real file bytes.
Neither establishes physical adequacy, temporal commitment or scientific
truth. Preserve an independently auditable pre-evaluation forecast record;
a later digest alone cannot prove when a prediction was issued.

Declare the five measurement-provenance questions explicitly: future
samples, outcome-derived wiring, fitting on evaluation data, evaluation
labels and post-selection. An affirmative answer can describe exploratory
evidence but cannot silently become prospective validation. Numerical
certificate tolerance is separate from the instrument/statistical budget.

P2 remains `not_admitted` until one accessible public-data choice resolves
the missing table, freezes all tolerances and resources, and passes the
state/clock/identifiability gate. Volts remains a historical exploratory record
of the earlier fixed-reference model/map pair, not an evaluation of the revised
generative candidate and not reserved data for a later redesign. The execution
plan owns when this parked branch resumes;
another fit to this same curve cannot supply missing apparatus evidence or
independent preparations. No apparatus purchase/acquisition is requested.
No admitted physical prediction has yet been tested.
