# Loaded finite-window observation of a prepared collective reference

<a id="loaded-window-design-declaration"></a>
## Declared question and candidate

This design admission follows the [prepared matrix-response theorem](../nodal/ORGANIZED_EFFECTIVE_LAW.md).
It asks whether a finite observation with an explicit passive sensor can
preserve a predictive consequence and support an independent physical test.
The [execution plan](FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) selects this
gate. No apparatus, measured response, source acquisition or fine-node
extension is supplied by the mathematical declaration below.

Keep the complete C5 law, signed forms U, circular phases Phi, held unit
capacities, fixed support, normalized structural clock tau and exact
`gamma=1/(1023*pi)` of the owner. Write `v=(1,-1,0,0,0)`,
`W_r=I+r*L^2`, `y=v^T*L*U`. Add one sensor state z with

\[
 W_r U'=(-LU+\gamma S(\Phi))/4+f-kv(y-z),\qquad
 \Phi'=\gamma LU/4,\qquad \tau_s z'=y-z.
 \tag{1}
\]

The sensor loads the form row through its actual current; it is not an
additive observation error. The selected observable y differs from the
earlier `v^T*U`: it is proportional to a physical differential voltage in
the declared electrical dictionary.

| Primitive | Fixed analytical declaration |
| --- | --- |
| Response family and control | Fixed `r in [1/100,1/10]`, compared separately with `r=0` |
| Full preparation | The earlier centered winding `w=+1` or `w=-1` reference, `||U(0)||_2+||theta(0)||_2<=delta=2^-26`; the same continuous perturbation lift and closing-edge offset |
| Sensor preparation | Independently bounded `|z(0)|<=delta_s=2^-26` |
| Input | `f=a*v` on `[0,T]`, zero afterwards; `a=1/16`, `T=1/1024` |
| Sensor law | `k=1/64`, `tau_s=T`; no resets, phase forcing or support events |
| Observation | `J=(1/T)*integral_0^T z(tau) d tau`, with separate additive error `|e|<=eta=2^-26` |
| Identity criterion | Every oriented phase gap remains within `rho=1/10` of its prepared `2*pi*w/5` during the pulse and later unforced evolution |
| Matrix-response criterion | Whole-preparation observed intervals for positive r and r=0 are disjoint with separation `>2^-21` |
| Phase-sensitive control | Compare with a passive linear reference having the same r, form/sensor initialization, input, clock and sensor but omitting the sine-current branch; determine whether permitted records must be disjoint or can overlap |
| Physical admission | Independent complete preparation, loaded transfer and matrices, delivery, clock and observation error/coverage under the same laboratory protocol |
| Calculation | Exact rational inequalities and certified exponential/trigonometric bounds; no trajectory, parameter scan, fitted response or physical dataset |

The phase-blind reference tests what this observable resolves after linear
calibration. It does not override independently established junction data.
An independently justified sine-current law would be additional evidence,
not something inferred from an unresolved pulse observation.

Source review concerns terrestrial methods and specifications only. Earlier
frozen protocols, source archives, responses and the separate endpoint-voltage
feasibility review remain unchanged. A conditional sensor equation is not a
manufacturer performance claim or a hardware implementation.
