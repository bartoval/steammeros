# Third-party notices

Code and models from other projects that end up in this library or its builds.

| Component | Used as | License |
|-|-|-|
| [SCNet](https://github.com/starrytong/SCNet), Copyright (c) 2024 starrytong | model weights (exported to ONNX); inference steps ported to C++ | MIT |
| [julius](https://github.com/adefossez/julius), Copyright 2020 Alexandre Défossez | `resample_frac` ported to `stems/resample.cpp` | MIT |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime), Copyright (c) Microsoft Corporation | API headers in `stems/external/onnxruntime/`; library loaded at run time, or built by `tools/build_ort.sh` and linked statically | MIT |
| [pocketfft](https://gitlab.mpcdf.mpg.de/mtr/pocketfft), Copyright (C) 2010-2018 Max-Planck-Society | header in `stems/external/pocketfft/`, compiled into the library | BSD-3-Clause |

ONNX Runtime bundles further libraries (abseil, protobuf, re2, Eigen, ...). Binaries that
include or ship it must also ship its `ThirdPartyNotices.txt`, found in the ONNX Runtime
release or in `third_party/onnxruntime/` after `tools/build_ort.sh`.

## MIT License (SCNet, julius, ONNX Runtime)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## BSD 3-Clause License (pocketfft)

Copyright (C) 2010-2018 Max-Planck-Society
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
