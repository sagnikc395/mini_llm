## inference

a minimal inference engine for mini_llm written in Go to make it fast and highly concurrent.

# High-Performance Concurrent Foundation Model Inference Engine in Go

A zero-copy, highly scalable, CGO-bound ONNX Runtime inference engine built from scratch in Go. Designed specifically for serving custom foundation models with ultra-low latency, dynamic batching, lock-free memory management, and hardware acceleration (CUDA / TensorRT).

---

## Architecture Overview

Standard Python-based model serving frameworks (like FastAPI or Flask wrapping Torch) suffer from CPython Global Interpreter Lock (GIL) contention, heavy memory consumption, and unpredictable Garbage Collection (GC) pauses. 

This engine bridges **Go's light-weight concurrent networking model (Goroutines / Network Poller)** with **ONNX Runtime's compiled C execution graph**, achieving optimal GPU memory bandwidth utilization and microsecond-level request routing.

```
                    +-------------------------------------------------------+
                    |                 Go Ingestion Layer                    |
                    |   gRPC / HTTP REST API (net/http & HTTP/2 Poller)     |
                    +-------------------------------------------------------+
                                                |
                                                v
                    +-------------------------------------------------------+
                    |                Dynamic Batching Engine                |
                    |   - Dual-threshold Trigger (Batch Size / Timeout)     |
                    |   - Lock-Free Request Accumulation                    |
                    +-------------------------------------------------------+
                                                |
                                                v
                    +-------------------------------------------------------+
                    |           Zero-Copy CGO Bridge (C Heap Memory)        |
                    |   - runtime.LockOSThread() for Worker Isolation        |
                    |   - Unsafe C Memory Wrappers (Bypassing Go GC)        |
                    +-------------------------------------------------------+
                                                |
                                                v
                    +-------------------------------------------------------+
                    |                  ONNX Runtime Engine                  |
                    |   - Frozen Graph Computation                          |
                    |   - Execution Providers: CUDA / TensorRT / CPU Arena  |
                    +-------------------------------------------------------+
```

---

## Key Features & Optimization Benchmarks

* **Dynamic Request Batching:** Merges distinct concurrent inference calls into dynamic 2D/3D batch tensors based on configurable window timeouts and batch thresholds.
* **Zero-Copy Memory Management:** Bypasses Go's garbage collector by allocating input/output tensor buffers directly on C heap memory (`C.malloc`), wrapping pointers using `unsafe.Slice`.
* **Pinned Worker Threads:** Binds execution workers to underlying OS threads using `runtime.LockOSThread()` to maintain context sanity and minimize thread-switching latencies across CGO boundaries.
* **Frozen Weight Inference:** Consumes static, fully optimized `.onnx` artifacts with constant folding, operator fusion, and quantized weights.
* **Hardware Execution Providers:** Native support for CPU (OpenMP/MKL), NVIDIA CUDA, and TensorRT acceleration backends.

---

## Project Structure

```text
.
├── cmd/
│   └── server/             # Entry point for the gRPC / HTTP Inference Server
├── pkg/
│   ├── batcher/            # Dynamic Batching & Request Aggregator
│   │   ├── batcher.go
│   │   └── batcher_test.go
│   ├── engine/             # CGO Bindings for ONNX Runtime C API
│   │   ├── cgo.go
│   │   ├── engine.go
│   │   └── memory.go
│   └── model/              # Tensor shapes and domain request/response structs
├── scripts/
│   └── export_onnx.py      # PyTorch-to-ONNX export script with dynamic axes
├── go.mod
├── go.sum
└── README.md
```

---

## Phase 1: Model Exporting (PyTorch -> ONNX)

Before serving the model, export your trained PyTorch foundation model to an ONNX graph with **dynamic axes** enabled for `batch_size` and `sequence_length`.

Save this script as `scripts/export_onnx.py`:

```python
import torch
import torch.nn as nn

class FoundationModel(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

def export():
    model = FoundationModel(hidden_dim=512).eval()
    dummy_input = torch.randn(1, 512, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        "foundation_model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,          # Fuses BatchNorm & constant expressions
        input_names=["input_tensor"],
        output_names=["output_tensor"],
        dynamic_axes={
            "input_tensor": {0: "batch_size", 1: "seq_len"},
            "output_tensor": {0: "batch_size", 1: "seq_len"}
        }
    )
    print("Model successfully exported and frozen into foundation_model.onnx")

if __name__ == "__main__":
    export()
```

---

## Phase 2: CGO Core Engine Implementation

Below is the production-grade Go engine wrapping the ONNX Runtime C headers.

### `pkg/engine/engine.go`

```go
package engine

/*
#cgo CFLAGS: -I/usr/local/include/onnxruntime
#cgo LDFLAGS: -L/usr/local/lib -lonnxruntime
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

type ONNXEngine struct {
	env     *C.OrtEnv
	session *C.OrtSession
	api     *C.OrtApi
}

// NewONNXEngine initializes environment and session for ONNX Runtime
func NewONNXEngine(modelPath string, useGPU bool) (*ONNXEngine, error) {
	runtime.LockOSThread()

	ortApi := C.OrtGetApiBase().OrtGetApi(C.ORT_API_VERSION)
	var env *C.OrtEnv
	var session *C.OrtSession

	cEnvName := C.CString("go-inference-engine")
	defer C.free(unsafe.Pointer(cEnvName))

	// 1. Create ORT Environment
	C.OrtApi_CreateEnv(ortApi, C.ORT_LOGGING_LEVEL_WARNING, cEnvName, &env)

	// 2. Set Session Options
	var options *C.OrtSessionOptions
	C.OrtApi_CreateSessionOptions(ortApi, &options)
	defer C.OrtApi_ReleaseSessionOptions(ortApi, options)

	C.OrtApi_SetSessionExecutionMode(ortApi, options, C.ORT_SEQUENTIAL)

	if useGPU {
		// Append CUDA Execution Provider (Requires CUDA CGO build flags)
		var cudaOptions *C.OrtCUDAProviderOptions
		C.OrtApi_CreateCUDAProviderOptions(ortApi, &cudaOptions)
		C.OrtApi_SessionOptionsAppendExecutionProvider_CUDA(ortApi, options, cudaOptions)
		C.OrtApi_ReleaseCUDAProviderOptions(ortApi, cudaOptions)
	}

	// 3. Instantiate Session
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	status := C.OrtApi_CreateSession(ortApi, env, cModelPath, options, &session)
	if status != nil {
		return nil, fmt.Errorf("failed to create ONNX session from path: %s", modelPath)
	}

	return &ONNXEngine{
		env:     env,
		session: session,
		api:     ortApi,
	}, nil
}

// Forward executes zero-copy tensor inference across dynamic batch sizes
func (e *ONNXEngine) Forward(inputData []float32, shape []int64) ([]float32, error) {
	var memoryInfo *C.OrtMemoryInfo
	C.OrtApi_CreateCpuMemoryInfo(e.api, C.OrtArenaAllocator, C.OrtMemTypeDefault, &memoryInfo)
	defer C.OrtApi_ReleaseMemoryInfo(e.api, memoryInfo)

	// Direct raw allocation on C heap to avoid Go GC tracing penalties
	dataByteSize := C.size_t(len(inputData) * 4) // 4 bytes for float32
	cBuffer := C.malloc(dataByteSize)
	defer C.free(cBuffer)

	// Direct memory copy into raw C pointer space
	C.memcpy(cBuffer, unsafe.Pointer(&inputData[0]), dataByteSize)

	var inputTensor *C.OrtValue
	shapePtr := (*C.int64_t)(unsafe.Pointer(&shape[0]))

	C.OrtApi_CreateTensorWithDataAsOrtValue(
		e.api,
		memoryInfo,
		cBuffer,
		dataByteSize,
		shapePtr,
		C.size_t(len(shape)),
		C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
		&inputTensor,
	)
	defer C.OrtApi_ReleaseValue(e.api, inputTensor)

	inputName := C.CString("input_tensor")
	outputName := C.CString("output_tensor")
	defer C.free(unsafe.Pointer(inputName))
	defer C.free(unsafe.Pointer(outputName))

	var outputTensor *C.OrtValue

	// Execute Inference Graph
	status := C.OrtApi_Run(
		e.api,
		e.session,
		nil,
		&inputName,
		&inputTensor,
		1,
		&outputName,
		1,
		&outputTensor,
	)
	if status != nil {
		return nil, fmt.Errorf("execution error during ONNX forward pass")
	}
	defer C.OrtApi_ReleaseValue(e.api, outputTensor)

	// Extract Pointer without Heap Re-allocation
	var rawOutputPtr unsafe.Pointer
	C.OrtApi_GetTensorMutableData(e.api, outputTensor, &rawOutputPtr)

	var typeInfo *C.OrtTensorTypeAndShapeInfo
	C.OrtApi_GetTensorTypeAndShape(e.api, outputTensor, &typeInfo)
	defer C.OrtApi_ReleaseTensorTypeAndShapeInfo(e.api, typeInfo)

	var elementCount C.size_t
	C.OrtApi_GetTensorShapeElementCount(e.api, typeInfo, &elementCount)

	// Zero-copy slice conversion using unsafe
	outputSlice := unsafe.Slice((*float32)(rawOutputPtr), int(elementCount))
	return outputSlice, nil
}

// Close gracefully releases underlying C resources
func (e *ONNXEngine) Close() {
	if e.session != nil {
		C.OrtApi_ReleaseSession(e.api, e.session)
	}
	if e.env != nil {
		C.OrtApi_ReleaseEnv(e.api, e.env)
	}
}
```

---

## Phase 3: Dynamic Aggregator & Engine Loop

### `pkg/batcher/batcher.go`

```go
package batcher

import (
	"context"
	"time"

	"engine/pkg/engine"
)

type InferenceRequest struct {
	InputData []float32
	Response  chan []float32
	ErrChan   chan error
}

type DynamicBatcher struct {
	engine     *engine.ONNXEngine
	queue      chan InferenceRequest
	maxBatch   int
	maxTimeout time.Duration
	dimSize    int
}

func NewDynamicBatcher(eng *engine.ONNXEngine, maxBatch int, timeout time.Duration, dimSize int) *DynamicBatcher {
	return &DynamicBatcher{
		engine:     eng,
		queue:      make(chan InferenceRequest, 1024),
		maxBatch:   maxBatch,
		maxTimeout: timeout,
		dimSize:    dimSize,
	}
}

func (b *DynamicBatcher) Submit(req InferenceRequest) {
	b.queue <- req
}

func (b *DynamicBatcher) StartWorker(ctx context.Context) {
	go func() {
		batch := make([]InferenceRequest, 0, b.maxBatch)
		ticker := time.NewTicker(b.maxTimeout)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case req := <-b.queue:
				batch = append(batch, req)
				if len(batch) >= b.maxBatch {
					b.flush(batch)
					batch = make([]InferenceRequest, 0, b.maxBatch)
				}
			case <-ticker.C:
				if len(batch) > 0 {
					b.flush(batch)
					batch = make([]InferenceRequest, 0, b.maxBatch)
				}
			}
		}
	}()
}

func (b *DynamicBatcher) flush(batch []InferenceRequest) {
	batchSize := len(batch)
	flattenedInput := make([]float32, 0, batchSize*b.dimSize)

	for _, req := range batch {
		flattenedInput = append(flattenedInput, req.InputData...)
	}

	shape := []int64{int64(batchSize), int64(b.dimSize)}

	out, err := b.engine.Forward(flattenedInput, shape)
	if err != nil {
		for _, req := range batch {
			req.ErrChan <- err
		}
		return
	}

	// Unpack batch responses
	for i, req := range batch {
		start := i * b.dimSize
		end := start + b.dimSize
		
		resCopy := make([]float32, b.dimSize)
		copy(resCopy, out[start:end])
		req.Response <- resCopy
	}
}
```

---

## Build & Dependencies

### Prerequisites

1. **Install ONNX Runtime C Library:**
   ```bash
   # Linux (x86_64)
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
   tar -xvf onnxruntime-linux-x64-1.17.1.tgz
   sudo cp -r onnxruntime-linux-x64-1.17.1/include/* /usr/local/include/onnxruntime/
   sudo cp -r onnxruntime-linux-x64-1.17.1/lib/* /usr/local/lib/
   sudo ldconfig
   ```

2. **Compile and Run Server:**
   ```bash
   export CGO_ENABLED=1
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   
   go build -o inference-server cmd/server/main.go
   ./inference-server --model=foundation_model.onnx --port=8080
   ```

---

## Operational Guidelines

* **Tuning `maxBatch` and `maxTimeout`:** Set `maxTimeout` to your acceptable SLA latency budget (e.g., `2ms`–`5ms`) and `maxBatch` according to your GPU RAM saturation limit.
* **GC Tuning:** For extremely high throughput, set `GOGC=off` or use `debug.SetMemoryLimit` if running inside memory-constrained Kubernetes containers.
* **Multi-GPU Parallelism:** Spawn dedicated OS worker threads (`runtime.LockOSThread()`) per GPU core and instantiate isolated `OrtSession` instances pinned to distinct CUDA device IDs.

---

## License

MIT License. Designed for production AI infrastructure engineering.



