#!/usr/bin/env python3
"""Benchmark the notebook model on Jetson Nano with Python 3.6 compatibility."""

from __future__ import print_function

import argparse
import json
import math
import os
import platform
import resource
import time

import numpy as np


INPUT_SHAPE = (1, 1, 64, 512)
OUTPUT_DIM = 512


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values):
    values = [float(v) for v in values]
    return {
        "count": len(values),
        "mean_ms": float(np.mean(values)),
        "median_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def strip_module_prefix(state):
    if not any(key.startswith("module.") for key in state):
        return state
    return {key.replace("module.", "", 1): value for key, value in state.items()}


def create_pytorch_runner(checkpoint_path):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)

    def conv1x1(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)

    class IBasicBlock(nn.Module):
        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(IBasicBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
            self.conv1 = conv3x3(inplanes, planes)
            self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
            self.prelu = nn.PReLU(planes)
            self.conv2 = conv3x3(planes, planes, stride)
            self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
            self.downsample = downsample

        def forward(self, value):
            identity = value
            output = self.bn1(value)
            output = self.conv1(output)
            output = self.bn2(output)
            output = self.prelu(output)
            output = self.conv2(output)
            output = self.bn3(output)
            if self.downsample is not None:
                identity = self.downsample(identity)
            return output + identity

    class IrisIResNet50MSFF(nn.Module):
        def __init__(self, num_features=512, dropout=0.35):
            super(IrisIResNet50MSFF, self).__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
            self.prelu = nn.PReLU(64)
            self.layer1 = self._make_layer(64, 3, 2)
            self.layer2 = self._make_layer(128, 4, 2)
            self.layer3 = self._make_layer(256, 14, 2)
            self.layer4 = self._make_layer(512, 3, 2)
            self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
            self.fusion_conv = nn.Conv2d(256, 256, 3, 2, 1, bias=False)
            self.fusion_bn = nn.BatchNorm2d(256, eps=1e-5)
            self.fusion_prelu = nn.PReLU(256)
            self.pool = nn.AdaptiveAvgPool2d((4, 8))
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(768 * 4 * 8, num_features)
            self.features = nn.BatchNorm1d(num_features, eps=1e-5)
            self.features.weight.requires_grad = False

        def _make_layer(self, planes, blocks, stride):
            downsample = None
            if stride != 1 or self.inplanes != planes:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes, stride),
                    nn.BatchNorm2d(planes, eps=1e-5),
                )
            layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes
            for _ in range(1, blocks):
                layers.append(IBasicBlock(self.inplanes, planes))
            return nn.Sequential(*layers)

        def forward(self, value):
            value = self.prelu(self.bn1(self.conv1(value)))
            value = self.layer1(value)
            value = self.layer2(value)
            layer3 = self.layer3(value)
            layer4 = self.layer4(layer3)
            deep = self.bn2(layer4)
            middle = self.fusion_prelu(self.fusion_bn(self.fusion_conv(layer3)))
            value = torch.cat((middle, deep), dim=1)
            value = self.pool(value)
            value = torch.flatten(value, 1)
            value = self.features(self.fc(self.dropout(value)))
            return functional.normalize(value, p=2, dim=1)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    dropout = float(config.get("dropout", config.get("dropout_rate", 0.35)))
    model = IrisIResNet50MSFF(dropout=dropout)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(strip_module_prefix(state), strict=True)
    model.cuda().eval()
    torch.backends.cudnn.benchmark = True

    class PyTorchRunner(object):
        backend = "pytorch_fp32"

        def __init__(self):
            self.model = model
            self.torch = torch
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats()

        def infer(self, array):
            host_start = time.perf_counter()
            tensor = self.torch.from_numpy(array).cuda(non_blocking=False)
            self.start_event.record()
            with self.torch.no_grad():
                output = self.model(tensor)
            self.end_event.record()
            result = output.detach().cpu().numpy()
            self.end_event.synchronize()
            compute_ms = float(self.start_event.elapsed_time(self.end_event))
            e2e_ms = (time.perf_counter() - host_start) * 1000.0
            return result, compute_ms, e2e_ms

        def memory_bytes(self):
            return int(self.torch.cuda.max_memory_allocated())

    return PyTorchRunner(), torch.__version__, str(torch.version.cuda)


def create_tensorrt_runner(engine_path, backend_name):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as handle:
        engine = trt.Runtime(logger).deserialize_cuda_engine(handle.read())
    if engine is None:
        raise RuntimeError("Could not deserialize TensorRT engine: " + engine_path)
    context = engine.create_execution_context()
    input_idx = next(i for i in range(engine.num_bindings) if engine.binding_is_input(i))
    output_idx = next(i for i in range(engine.num_bindings) if not engine.binding_is_input(i))
    if -1 in tuple(engine.get_binding_shape(input_idx)):
        if not context.set_binding_shape(input_idx, INPUT_SHAPE):
            raise RuntimeError("Could not set TensorRT input binding shape")
    input_shape = tuple(context.get_binding_shape(input_idx))
    output_shape = tuple(context.get_binding_shape(output_idx))
    if input_shape != INPUT_SHAPE or int(np.prod(output_shape)) != OUTPUT_DIM:
        raise RuntimeError("Unexpected TensorRT bindings: %r -> %r" % (input_shape, output_shape))
    input_dtype = trt.nptype(engine.get_binding_dtype(input_idx))
    output_dtype = trt.nptype(engine.get_binding_dtype(output_idx))
    host_input = cuda.pagelocked_empty(int(np.prod(input_shape)), input_dtype)
    host_output = cuda.pagelocked_empty(int(np.prod(output_shape)), output_dtype)
    device_input = cuda.mem_alloc(host_input.nbytes)
    device_output = cuda.mem_alloc(host_output.nbytes)
    bindings = [0] * engine.num_bindings
    bindings[input_idx] = int(device_input)
    bindings[output_idx] = int(device_output)
    stream = cuda.Stream()
    start_event = cuda.Event()
    end_event = cuda.Event()
    allocated_bytes = int(host_input.nbytes + host_output.nbytes + host_input.nbytes + host_output.nbytes)

    class TensorRTRunner(object):
        backend = backend_name

        def infer(self, array):
            host_start = time.perf_counter()
            np.copyto(host_input, array.astype(input_dtype, copy=False).ravel())
            cuda.memcpy_htod_async(device_input, host_input, stream)
            start_event.record(stream)
            ok = context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            end_event.record(stream)
            if not ok:
                raise RuntimeError("TensorRT execution failed")
            cuda.memcpy_dtoh_async(host_output, device_output, stream)
            stream.synchronize()
            compute_ms = float(start_event.time_till(end_event))
            e2e_ms = (time.perf_counter() - host_start) * 1000.0
            result = np.asarray(host_output, dtype=np.float32).reshape(output_shape).copy()
            return result, compute_ms, e2e_ms

        def memory_bytes(self):
            return allocated_bytes

    return TensorRTRunner(), trt.__version__, "10.2"


def validate_output(output):
    flat = np.asarray(output, dtype=np.float32).reshape(-1)
    if flat.size != OUTPUT_DIM or not np.all(np.isfinite(flat)):
        raise RuntimeError("Invalid embedding output")
    norm = float(np.linalg.norm(flat))
    if norm < 1e-6:
        raise RuntimeError("Zero embedding output")
    return norm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["pytorch_fp32", "tensorrt_fp32", "tensorrt_fp16"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--engine")
    parser.add_argument("--artifact")
    parser.add_argument("--output", required=True)
    parser.add_argument("--parity-output", required=True)
    parser.add_argument("--inputs-npy")
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument("--trial-seconds", type=float, default=120.0)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.backend == "pytorch_fp32":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for PyTorch")
        runner, runtime_version, cuda_version = create_pytorch_runner(args.checkpoint)
        artifact = args.artifact or args.checkpoint
    else:
        if not args.engine:
            raise SystemExit("--engine is required for TensorRT")
        runner, runtime_version, cuda_version = create_tensorrt_runner(args.engine, args.backend)
        artifact = args.artifact or args.engine

    if args.inputs_npy:
        inputs = np.load(args.inputs_npy).astype(np.float32, copy=False)
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != INPUT_SHAPE[1:]:
            raise RuntimeError("Unexpected input tensor array shape: %r" % (inputs.shape,))
        inputs = inputs.reshape((inputs.shape[0],) + INPUT_SHAPE)
    else:
        random = np.random.RandomState(args.seed)
        inputs = random.normal(0.0, 1.0, size=(32,) + INPUT_SHAPE[1:]).astype(np.float32)
        inputs = inputs.reshape((32,) + INPUT_SHAPE)

    parity = []
    norms = []
    for array in inputs:
        output, _, _ = runner.infer(array)
        parity.append(np.asarray(output, dtype=np.float32).reshape(-1))
        norms.append(validate_output(output))
    parity = np.stack(parity, axis=0)
    np.save(args.parity_output, parity)

    warmup_end = time.time() + args.warmup_seconds
    warmup_count = 0
    while time.time() < warmup_end:
        runner.infer(inputs[warmup_count % len(inputs)])
        warmup_count += 1

    trials = []
    all_compute = []
    all_e2e = []
    for trial_index in range(args.trials):
        compute_values = []
        e2e_values = []
        trial_start = time.time()
        count = 0
        while time.time() - trial_start < args.trial_seconds:
            output, compute_ms, e2e_ms = runner.infer(inputs[count % len(inputs)])
            validate_output(output)
            compute_values.append(compute_ms)
            e2e_values.append(e2e_ms)
            count += 1
        elapsed = time.time() - trial_start
        all_compute.extend(compute_values)
        all_e2e.extend(e2e_values)
        trials.append({
            "trial": trial_index + 1,
            "elapsed_seconds": elapsed,
            "inferences": count,
            "throughput_fps": count / elapsed,
            "compute": summarize(compute_values),
            "transfer_included": summarize(e2e_values),
        })
        print("%s trial %d/%d: %.3f ms median, %.3f FPS" % (
            args.backend, trial_index + 1, args.trials,
            trials[-1]["transfer_included"]["median_ms"], trials[-1]["throughput_fps"]), flush=True)

    result = {
        "schema_version": 1,
        "backend": args.backend,
        "runtime_version": runtime_version,
        "cuda_version": cuda_version,
        "python_version": platform.python_version(),
        "input_shape": list(INPUT_SHAPE),
        "output_dim": OUTPUT_DIM,
        "artifact": os.path.abspath(artifact),
        "artifact_bytes": os.path.getsize(artifact),
        "warmup_seconds": args.warmup_seconds,
        "warmup_inferences": warmup_count,
        "trial_seconds": args.trial_seconds,
        "trial_count": args.trials,
        "trials": trials,
        "overall_compute": summarize(all_compute),
        "overall_transfer_included": summarize(all_e2e),
        "throughput_fps": len(all_e2e) / (sum(all_e2e) / 1000.0),
        "embedding_norm_min": float(np.min(norms)),
        "embedding_norm_max": float(np.max(norms)),
        "process_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "backend_memory_bytes": int(runner.memory_bytes()),
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
