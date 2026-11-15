package stems

import "fmt"

// Pinned upstream releases. NVIDIA wheels are ZIP containers of native runtimes;
// Prepare extracts their libraries/licenses directly, without invoking Python/pip.
const ortRelease = "https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/"
const modelRelease = "https://github.com/bartoval/steammeros/releases/download/models-v1/"

func preparedAssets(opts PrepareOptions, osName, arch string) ([]asset, string, error) {
	if opts.Fast && !opts.CUDA {
		return nil, "", fmt.Errorf("Fast requires CUDA")
	}
	var archive, checksum string
	library := "libonnxruntime.so.1.31.1"
	switch {
	case opts.CUDA && osName == "linux" && arch == "amd64":
		archive = "onnxruntime-linux-x64-gpu_cuda13-1.31.1.tgz"
		checksum = "b2146862e691f7e338f5aff9c7a9a5567c2ca57cf01442a1adb3a13bf62d9095"
	case opts.CUDA:
		return nil, "", fmt.Errorf("prepared CUDA runtimes support linux/amd64, not %s/%s", osName, arch)
	case osName == "linux" && arch == "amd64":
		archive = "onnxruntime-linux-x64-1.31.1.tgz"
		checksum = "c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9"
	case osName == "linux" && arch == "arm64":
		archive = "onnxruntime-linux-aarch64-1.31.1.tgz"
		checksum = "5317505e2da99be65f8e2775ba3c2afd6d2961fd78aba1bd8f56159f48820c78"
	case osName == "darwin" && arch == "arm64":
		archive = "onnxruntime-osx-arm64-1.31.1.tgz"
		checksum = "2e1344a1926ee50eb25b38153cb6992c3567515aef0e677a1a520a1798896d6f"
		library = "libonnxruntime.1.31.1.dylib"
	default:
		return nil, "", fmt.Errorf("no prepared runtime for %s/%s", osName, arch)
	}
	model := asset{URL: modelRelease + "scnet-large.onnx", SHA256: "efeb7d69abe78e456ad8fa7609d150cc67f36bff5ce2e78f7743870a56c66432", Format: "file", Target: "scnet-large.onnx"}
	if opts.Fast {
		model = asset{URL: modelRelease + "scnet-large-cuda.onnx", SHA256: "e5d3c9a07334109a8824a177dcf8b9e96cbf4630c7f7e3aff2a57d03e1041eaa", Format: "file", Target: "scnet-large-cuda.onnx"}
	}
	assets := []asset{model, {URL: ortRelease + archive, SHA256: checksum, Format: "tgz"},
		{URL: "https://raw.githubusercontent.com/bartoval/steammeros/1305997679c3688100611911820cbb88bea18a36/LICENSE", SHA256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30", Format: "file", Target: "licenses/steammeros/LICENSE"}}
	if opts.CUDA {
		assets = append(assets, cudaAssets...)
	}
	if opts.Fast {
		assets = append(assets, asset{URL: modelRelease + "libstems_cuda_ops-linux-x64.so", SHA256: "f207a76ded8c3ed995d42689e0f31a4ff8972a80791795d8eff2a6dbc199eb57", Format: "file", Target: "lib/libstems_cuda_ops.so"})
	}
	return assets, library, nil
}

var cudaAssets = []asset{
	// nvidia-cuda-runtime 13.2.86
	{URL: "https://files.pythonhosted.org/packages/70/c6/8f8c5e893d183082ea78f973aebc7ebc845ee219d68bddac73b0f62490fe/nvidia_cuda_runtime-13.2.86-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl", SHA256: "3498247680dfddce31bad102af7e3e26fc795342a3cac64722bb0fa95e7f60f0", Format: "zip"},
	// nvidia-cuda-nvrtc 13.2.86
	{URL: "https://files.pythonhosted.org/packages/1e/39/c58835ec8392a91b8d7bad83760eafc3085f64788b722ec2b44a27b3ca3e/nvidia_cuda_nvrtc-13.2.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl", SHA256: "8111c5130e967b78222ea2124145f8efe9ce201b752b2579a2513100a53be1c5", Format: "zip"},
	// nvidia-cublas 13.4.1.3
	{URL: "https://files.pythonhosted.org/packages/e1/97/d483a41da34df5c88727c57319576967f6d7befba5028d70928d2999635e/nvidia_cublas-13.4.1.3-py3-none-manylinux_2_27_x86_64.whl", SHA256: "a5917cb642cfb07f4a0231ee16aec7e2e0357d8cb8b1b78a0c9c68dba3e6afc7", Format: "zip"},
	// nvidia-curand 10.4.5.25
	{URL: "https://files.pythonhosted.org/packages/12/63/636e5dc9afb2959bfc0042a357d5567358df829b66c96d9fc36d77dc0596/nvidia_curand-10.4.5.25-py3-none-manylinux_2_27_x86_64.whl", SHA256: "22bb226a857e5888fe6b54ae465fc4371296bcff13a35f79e19691bcf454f529", Format: "zip"},
	// nvidia-cudnn-cu13 9.26.0.51
	{URL: "https://files.pythonhosted.org/packages/ac/1b/ea72dd62f26ce3e0a7870b85148b75a5987507629b712deebeb87f7cce9a/nvidia_cudnn_cu13-9.26.0.51-py3-none-manylinux_2_27_x86_64.whl", SHA256: "9c976d539786698c71d6bcdbe2053b7485fae062ff7e1215d2f2ff67b1a2149a", Format: "zip"},
}

const scnetLicense = `MIT License

Copyright (c) 2024 starrytong

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
`
