"""Shared runtime checks for conditional IEEE binary64 numeric results."""

import math
import sys


def uses_ieee_binary64_rounding() -> bool:
    """Check the required format and basic nearest-even/subnormal behavior.

    These probes are an execution precondition, not an exhaustive hardware
    conformance proof or a guarantee about custom/native numeric kernels.
    """
    smallest = math.ulp(0.0)
    try:
        return bool(
            sys.float_info.radix == 2
            and sys.float_info.mant_dig == 53
            and sys.float_info.min_exp == -1021
            and sys.float_info.max_exp == 1024
            and sys.float_info.rounds == 1
            and float.__getformat__("double").startswith("IEEE")
            and smallest == float.fromhex("0x0.0000000000001p-1022")
            and 0.5 * smallest == 0.0
            and 0.5 * (3.0 * smallest) == 2.0 * smallest
            and 0.5 * (-3.0 * smallest) == -2.0 * smallest
        )
    except BaseException:
        return False
