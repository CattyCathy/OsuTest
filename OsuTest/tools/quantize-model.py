"""Quantises the Beat This! ONNX export to int8 for shipping.

Run once when the model is updated; the result is what the player loads.

    python tools/quantize-model.py <float-model.onnx> [<int8-model.onnx>]

Measured on the model this was written for (beat-this-final0), against a beatmap's own timing points:

    float      78.3 MB   65.0s inference for 178s of audio   (36% of realtime)
    int8       20.9 MB   45.6s inference for 178s of audio   (26% of realtime)

and the same beats at the same metrical levels, so the smaller model is not a trade. The gain is smaller than
int8 usually gives because the cost of this model is dominated by attention rather than the matrix multiplies
that quantisation speeds up, but 3.7x smaller on disk and 1.4x faster for nothing is still worth taking.
"""

import os
import sys
import time

from onnxruntime.quantization import QuantType, quantize_dynamic


def main(argv):
    if not argv:
        sys.exit(__doc__)

    source = argv[0]
    target = argv[1] if len(argv) > 1 else source.replace(".onnx", "-int8.onnx")

    if not os.path.exists(source):
        sys.exit(f"no model at {source}")

    before = os.path.getsize(source)
    start = time.time()

    quantize_dynamic(model_input=source, model_output=target, weight_type=QuantType.QInt8)

    after = os.path.getsize(target)
    print(f"float:     {before / 1024 / 1024:7.1f} MB")
    print(f"quantised: {after / 1024 / 1024:7.1f} MB  ({after / before:.0%})")
    print(f"took {time.time() - start:.0f}s -> {target}")


if __name__ == "__main__":
    main(sys.argv[1:])
