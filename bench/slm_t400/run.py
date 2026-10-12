"""SLM rearrangement solver benchmark on this machine's GPU (the lab's NVIDIA T400).

Times this checkout's solver, and commit de73d721 (the LPI solver before the ghost-trap change; its solver.py is
solver_de73d721.py here, byte for byte) for the sequence comparison, on run 14 of 2026-10-10 (35 sources, its 16-site target) and on the
lab's current central 3x3 target. bin/slm_t400_bench.bat runs it; close the SLM task console first. Everything
lands in bench/results/slm_t400_<host>_<time>/ and a .zip of that folder:

  result.json   every section below
  log.txt       the console output
  clocks.csv    nvidia-smi every 100 ms for the whole run (SM clock, P-state, utilisation, power)
  nvidia_smi.txt  nvidia-smi -q before the run

Sections:
  calib    FP32 and FP64 FMA rates, device-memory copy bandwidth, pinned 1.3 MB host<->device copies
  gemm     the analysis product through cuBLAS (the production call and its alternatives) and cuBLAS-free
           kernels, K = 8 ... 64 bands
  kernels  one evaluation (encode, correct, measure) and each of its kernels, the ghost watch, track kernels,
           on run 14's removal, next and transport rosters
  maps     one map of every schedule kind as its captured graph(s), on run 14 and on the 3x3 target
  plane    the refined last map's capped full-plane iterations and the last map's stray measurement
  ramp     a transport map launched 60 times right after 2 s idle: per-launch device time against elapsed time
           (does the GPU start slow and speed up within a sequence?)
  seq      this checkout against de73d721 in one process, alternating: the Fixed task path from the occupancy
           list (plan + compute that samples itself) to the first and the last map, and each map's solve time
  gap      the lab's timing: 3 s of idle GPU (between shots and the exposure, only the solver's own heater running)
           before the same Fixed path, for this checkout and de73d721, against a warm GPU; with a simulated 60 Hz
           playback of the maps as they get ready
"""
import argparse
import contextlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SHAPE = (1024, 1272)
LOG = []

KERNELS = r'''
#include <cuda_fp16.h>
extern "C" __global__ void fma_peak(float* out,int iters){
 float a[8];for(int j=0;j<8;j++)a[j]=threadIdx.x*1e-3f+j;
 for(int i=0;i<iters;i++){
  #pragma unroll
  for(int j=0;j<8;j++)a[j]=fmaf(a[j],1.0000001f,-1e-7f);}
 float s=0;for(int j=0;j<8;j++)s+=a[j];if(s==12345.f)out[0]=s;
}
extern "C" __global__ void dfma_peak(double* out,int iters){
 double a[4];for(int j=0;j<4;j++)a[j]=threadIdx.x*1e-3+j;
 for(int i=0;i<iters;i++){
  #pragma unroll
  for(int j=0;j<4;j++)a[j]=fma(a[j],1.0000001,-1e-7);}
 double s=0;for(int j=0;j<4;j++)s+=a[j];if(s==12345.)out[0]=s;
}
extern "C" __global__ void copy_u32(const unsigned int* __restrict__ a,unsigned int* __restrict__ b,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=a[i];
}
#ifndef AR
#define AR 4
#endif
#ifndef KC
#define KC 8
#endif
extern "C" __global__ void analysis_warp(const half* __restrict__ field,const float* __restrict__ forward,
 float* __restrict__ out,int rows,int C,int K){
 int lane=threadIdx.x&31,row0=((blockIdx.x*blockDim.x+threadIdx.x)>>5)*AR;if(row0>=rows)return;
 for(int k0=0;k0<K;k0+=KC){
  float acc[AR][KC];
  #pragma unroll
  for(int r=0;r<AR;r++)
   #pragma unroll
   for(int k=0;k<KC;k++)acc[r][k]=0;
  for(int c=lane;c<C;c+=32){
   float v[AR];
   #pragma unroll
   for(int r=0;r<AR;r++)v[r]=__half2float(field[(row0+r)*C+c]);
   const float4* f=(const float4*)(forward+c*K+k0);
   #pragma unroll
   for(int k=0;k<KC;k+=4)if(k0+k<K){float4 w=f[k/4];
    #pragma unroll
    for(int r=0;r<AR;r++){acc[r][k]+=v[r]*w.x;acc[r][k+1]+=v[r]*w.y;acc[r][k+2]+=v[r]*w.z;acc[r][k+3]+=v[r]*w.w;}}
  }
  #pragma unroll
  for(int r=0;r<AR;r++)
   #pragma unroll
   for(int k=0;k<KC;k++)if(k0+k<K){float a=acc[r][k];
    #pragma unroll
    for(int o=16;o;o>>=1)a+=__shfl_xor_sync(0xffffffffu,a,o);
    if(lane==((r*KC+k)&31))out[(row0+r)*K+k0+k]=a;}
 }
}
'''


def log(*args):
    line = " ".join(str(a) for a in args)
    LOG.append(line)
    print(line, flush=True)


class Clocks:
    """nvidia-smi sampled back to back on a thread (each call a complete row, about 5-10 per second)."""
    QUERY = ("timestamp,clocks.sm,clocks.max.sm,clocks.mem,pstate,utilization.gpu,power.draw,temperature.gpu,"
             "clocks_throttle_reasons.active")

    def __init__(self, smi, path):
        self.smi, self.path, self.rows, self.stop = smi, path, [], threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            done = subprocess.run([self.smi, f"--query-gpu={self.QUERY}", "--format=csv,noheader"],
                                  capture_output=True, text=True)
            self.rows.append(done.stdout.strip())

    def close(self):
        self.stop.set()
        self.thread.join()
        with open(self.path, "w") as f:
            f.write("\n".join([self.QUERY, *self.rows]) + "\n")


def load_solvers():
    for src in sorted(os.listdir(os.path.join(ROOT, "packages"))):
        path = os.path.join(ROOT, "packages", src, "src")
        if os.path.isdir(path):
            sys.path.insert(0, path)
    sys.path.insert(0, ROOT)
    import zou_lab_control
    from zlc_atom.devices.slm import solver as new
    log("zou_lab_control:", zou_lab_control.__file__)
    log("solver new:", new.__file__)
    spec = importlib.util.spec_from_file_location("zlc_atom.devices.slm._bench_lpi", os.path.join(HERE, "solver_de73d721.py"))
    lpi = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lpi
    spec.loader.exec_module(lpi)
    log("solver lpi:", lpi.__file__)
    return {"new": new, "lpi": lpi}


def git_head():
    """The checked-out commit, read from .git without needing git on the PATH."""
    git = os.path.join(ROOT, ".git")
    try:
        with open(os.path.join(git, "HEAD")) as f:
            head = f.read().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        path = os.path.join(git, *ref.split("/"))
        if os.path.exists(path):
            with open(path) as f:
                return f"{ref} {f.read().strip()}"
        with open(os.path.join(git, "packed-refs")) as f:
            for line in f:
                if line.rstrip().endswith(" " + ref):
                    return f"{ref} {line.split()[0]}"
        return ref
    except OSError as error:
        return f"unknown ({error})"


@contextlib.contextmanager
def active(solver, prepared):
    """Pause the workspace's idle GPU heater while timing (the product does the same while it computes)."""
    with solver._rearrangement_gpu_active(prepared["gpu"]["keeper"]):
        yield


def graph_timer(cp, stream, launch, reps=10, trials=5, warm=1):
    """Per-launch microseconds of `launch` captured `reps` times in one CUDA graph: (min, median)."""
    with stream:
        launch()
        stream.synchronize()
        stream.begin_capture(cp.cuda.runtime.streamCaptureModeThreadLocal)
        try:
            for _ in range(reps):
                launch()
        finally:
            graph = stream.end_capture()
        graph.upload(stream)
        a, b = cp.cuda.Event(), cp.cuda.Event()
        out = []
        for t in range(trials + warm):
            a.record(stream)
            graph.launch(stream)
            b.record(stream)
            b.synchronize()
            if t >= warm:
                out.append(cp.cuda.get_elapsed_time(a, b) * 1000 / reps)
    return float(np.min(out)), float(np.median(out))


def direct_timer(cp, stream, launch, reps=3, trials=5, warm=1):
    a, b = cp.cuda.Event(), cp.cuda.Event()
    out = []
    for t in range(trials + warm):
        a.record(stream)
        for _ in range(reps):
            launch()
        b.record(stream)
        b.synchronize()
        if t >= warm:
            out.append(cp.cuda.get_elapsed_time(a, b) * 1000 / reps)
    return float(np.min(out)), float(np.median(out))


def scenario(name, solver):
    """Run 14 of 2026-10-10 (35 sources -> 16 sites, 15 maps); real14c3: the same load onto the central 3x3 of the
    source grid (the lab's current target); real14p13: 13 atoms on run 14's 16-site target (a partial fill)."""
    z = np.load(os.path.join(HERE, "real14.npz"))
    meta = json.loads(str(z["metadata"]))
    sc = meta["science_context"]
    _, amplitude = solver.science_pupil_fields(SHAPE, sc["pupil"])
    operator = np.asarray(solver.science_operator_wavefront(SHAPE, sc["pupil"], sc["operator_metadata"]))
    s = dict(name=name, source=z["source_yx"], target=z["target_yx"], pattern=z["source_pattern_phase"],
             occupied=np.flatnonzero(z["before_occupied"]), minimum_separation=float(meta["minimum_separation"]),
             target_intensities=np.asarray(z["target_requested_intensities"], np.float32),
             frames=int(meta["motion_frames"]), pupil_amplitude=amplitude, operator=operator,
             phase_center_yx=tuple(sc["pupil"]["center_xy"])[::-1])
    if name == "real14c3":
        src = np.asarray(s["source"])
        ys, xs = np.unique(src[:, 0]), np.unique(src[:, 1])
        y0, x0 = (len(ys) - 2) // 2, (len(xs) - 2) // 2
        yy, xx = np.meshgrid(ys[y0:y0 + 3], xs[x0:x0 + 3], indexing="ij")
        s.update(target=np.column_stack((yy.ravel(), xx.ravel())).astype(src.dtype),
                 target_intensities=np.ones(9, np.float32))
    elif name == "real14p13":
        s.update(occupied=np.sort(np.random.default_rng(81013).choice(len(s["source"]), 13, replace=False)), frames=16)
    return s


def prepare(solver, s, lpi):
    phase = solver.compose_science_phase(s["pattern"], s["operator"])
    kw = dict(shape_yx=SHAPE, pupil_amplitude=s["pupil_amplitude"], pupil_phase=-s["operator"],
              phase_center_yx=s["phase_center_yx"], target_intensities=s["target_intensities"],
              minimum_separation=s["minimum_separation"], endpoint_data={"source_phase": phase})
    if lpi:
        kw.update(method="lpi", maximum_motion_frames=max(s["frames"] or 16, 1))
    return solver.prepare_rearrangement(s["source"], s["target"], **kw)


def rosters(solver, prepared, s):
    """Each map role's real roster: (points, retained count, dark count)."""
    plan = solver.plan_rearrangement(prepared, s["occupied"])
    sampled = solver.sample_rearrangement(prepared, plan, motion_frames=s["frames"])
    sel = np.asarray(plan["source_indices"])
    dark = np.setdiff1d(np.arange(len(prepared["source_yx"])), sel)
    sites = np.asarray(sampled["sites_yx"])
    return {"release": (np.concatenate((sites[0][sel], prepared["source_yx"][dark])), len(sel), len(dark)),
            "next": (np.concatenate((sites[1][sel], prepared["source_yx"][dark])), len(sel), len(dark)),
            "transport": (sites[len(sites) // 2][sel], len(sel), 0),
            "last": (sites[-1][sel], len(sel), 0)}


def bind(solver, gpu, points, kept, dark, scalars=(.32, 1., 1.1)):
    """Bind one map record (unit goals and rotors, dark points capped at zero); returns (band, record bytes)."""
    bands, signed_y, frequencies, counts, _ = solver._rearrangement_bind(np.asarray(points, np.float64)[None], gpu["shape"])
    band = (int(counts[0]) + 7) // 8 * 8
    row = np.zeros(gpu["record"].size, np.uint8)
    views = solver._rearrangement_record_views(row, gpu["actual"].size, gpu["capacity"])
    solver._rearrangement_record(views, bands[0], signed_y[0], frequencies, band)
    views["goal"][:kept] = 1
    views["rotor"][:kept] = 1
    views["cap"][:dark] = 0
    views["gate"][:dark] = 1e-3
    views["scalars"][:] = scalars
    views["meta"][:] = (kept, dark)
    gpu["record"].set(row, stream=gpu["stream"])
    return band, row


def start_state(gpu, kept):
    c = np.zeros(gpu["actual"].size, np.complex64)
    c[:kept] = np.exp(1j * np.linspace(0, 6, kept)) / np.sqrt(kept)
    gpu["track_c"].set(c, stream=gpu["stream"])
    gpu["track_cancel"].fill(0)
    values = np.zeros(gpu["actual"].size + 1, np.complex64)
    values[:len(c)] = c
    values[-1] = 1
    gpu["coefficients"].set(values, stream=gpu["stream"])


def kernel_launches(m, gpu, band):
    """Name -> launch of one evaluation and of each of its kernels, with the solver's own arguments."""
    K_, cb, gemm = gpu["kernels"], gpu["cublas"], gpu["gemm"]
    h, w = gpu["shape"]
    P, sites = gpu["padded"], gpu["actual"].size
    alpha, beta = gpu["alpha"].ctypes.data, gpu["beta"].ctypes.data
    L = {f"evaluation_{op}": (lambda op=op: m._rearrangement_propagate(gpu, band, op)) for op in ("encode", "correct", "measure")}
    L[f"evaluation_correct(band+{m._TRACK_WATCH_BANDS})"] = lambda: m._rearrangement_propagate(gpu, band + m._TRACK_WATCH_BANDS, "correct")
    for mode, op in enumerate(("measure", "encode", "correct")):
        L[f"evaluate_kernel_{op}"] = lambda mode=mode: K_["evaluate"](
            ((P + 127) // 128, (h + 7) // 8), (128,),
            (gpu["coefficients"], gpu["band_start"], gpu["band_point"], gpu["point_roots"], gpu["roots"], gpu["amplitude"],
             gpu["incident_turns"], gpu["codes"], gpu["field"], np.float32(2 / gpu["pupil_energy"]),
             *map(np.int32, (mode, sites, h, w, P, band))), shared_mem=8 * band * 8)
    L["analysis_gemm(cuBLAS)"] = lambda: gemm(gpu["handle"], 1, 1, 2 * h, band, 2 * P, alpha, gpu["field"].data.ptr, 2, 2 * P,
                                              gpu["forward"].data.ptr, 2, band, beta, gpu["projected"].data.ptr, 0, 2 * h,
                                              cb.CUBLAS_COMPUTE_32F, cb.CUBLAS_GEMM_DEFAULT)
    L["project_points"] = lambda: K_["project_points"]((sites,), (256,), (gpu["projected"], gpu["point_band"],
                                                                         gpu["point_roots"], gpu["actual"], np.int32(h)))
    L["roots(per map)"] = lambda: m._rearrangement_roots(gpu, band)
    L["track_sweep"] = lambda: K_["track_sweep"]((1,), (128,), (
        gpu["actual"], gpu["goal"], gpu["rotor"], gpu["cap"], gpu["gate"], gpu["scalars"], gpu["meta"], gpu["track_c"],
        gpu["track_cancel"], gpu["track_accepted"], gpu["coefficients"], np.int32(sites), np.float32(.8)))
    cap, detection = m._TRACK_WATCH_LIMITS["transport"][:2]
    L["track_residual(Anderson)"] = lambda: K_["track_residual"]((1,), (128,), (
        gpu["actual"], gpu["goal"], gpu["cap"], gpu["gate"], gpu["scalars"], gpu["meta"], gpu["coefficients"],
        np.int32(sites), np.int32(m._TRACK_WATCH), *np.sqrt(np.float32((cap, m._TRACK_WATCH_GATE[0]))), np.int32(3),
        gpu["track_history"], gpu["track_steps"]))
    # The watch rewrites the record's band CSR and its work grows with the local maxima above the detection, so it
    # runs on a frozen copy of the state the setup encode left, with the record restored before each launch.
    frozen = dict(gpu, field=gpu["field"].copy(), actual=gpu["actual"].copy())
    record = gpu["record"].copy()
    L["watch"] = lambda: (gpu["record"].__setitem__(Ellipsis, record), m._rearrangement_watch(frozen, band, detection))
    return L


# ---------------------------------------------------------------- sections
def section_calib(cp):
    stream = cp.cuda.Stream(non_blocking=True)
    mod = cp.RawModule(code=KERNELS)
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    out = dict(device=dict(name=props["name"].decode(), sm_count=props["multiProcessorCount"], clock_khz=props["clockRate"],
                           memory_clock_khz=props["memoryClockRate"], bus_width_bits=props["memoryBusWidth"],
                           l2_bytes=props["l2CacheSize"], compute_capability=f'{props["major"]}.{props["minor"]}',
                           shared_per_block_optin=props["sharedMemPerBlockOptin"]))
    threads = props["multiProcessorCount"] * 2048
    dummy = cp.zeros(1, cp.float64)
    t = graph_timer(cp, stream, lambda: mod.get_function("fma_peak")((threads // 256,), (256,), (dummy, np.int32(512))), reps=5)
    out["fp32_fma_lane_ops_per_s"] = threads * 512 * 8 / (t[0] * 1e-6)
    t = graph_timer(cp, stream, lambda: mod.get_function("dfma_peak")((threads // 256,), (256,), (dummy, np.int32(64))), reps=3)
    out["fp64_fma_lane_ops_per_s"] = threads * 64 * 4 / (t[0] * 1e-6)
    big, big2 = cp.zeros(16 << 20, cp.uint32), cp.zeros(16 << 20, cp.uint32)
    t = graph_timer(cp, stream, lambda: mod.get_function("copy_u32")(((big.size + 255) // 256,), (256,), (big, big2, np.int32(big.size))), reps=5)
    out["device_copy_bytes_per_s"] = 2 * big.nbytes / (t[0] * 1e-6)
    del big, big2
    area = SHAPE[0] * SHAPE[1]
    pinned = cp.cuda.alloc_pinned_memory(area)
    host = np.frombuffer(pinned, np.uint8, count=area)
    dev, dst = cp.zeros(area, cp.uint8), cp.zeros(area, cp.uint8)
    rt = cp.cuda.runtime
    out["d2h_codes_1p3MB_us"] = direct_timer(cp, stream, lambda: rt.memcpyAsync(host.ctypes.data, dev.data.ptr, area, rt.memcpyDeviceToHost, stream.ptr), reps=10)
    out["h2d_codes_1p3MB_us"] = direct_timer(cp, stream, lambda: rt.memcpyAsync(dev.data.ptr, host.ctypes.data, area, rt.memcpyHostToDevice, stream.ptr), reps=10)
    out["d2d_codes_1p3MB_us"] = direct_timer(cp, stream, lambda: rt.memcpyAsync(dst.data.ptr, dev.data.ptr, area, rt.memcpyDeviceToDevice, stream.ptr), reps=10)
    log("calib", json.dumps({k: (v if not isinstance(v, float) else float("%.4g" % v)) for k, v in out.items()}))
    return out


def section_gemm(cp, m, prepared):
    gpu = prepared["gpu"]
    stream, (h, w), P = gpu["stream"], gpu["shape"], gpu["padded"]
    cb, gemm = gpu["cublas"], gpu["gemm"]
    one, zero = np.asarray(1, np.float32), np.asarray(0, np.float32)
    out = {}
    with stream:
        rng = np.random.default_rng(0)
        field = cp.asarray(rng.normal(size=(2 * h, 2 * P)).astype(np.float16))
        field32 = field.astype(cp.float32)
        warp = {(ar, kc): cp.RawModule(code=f"#define AR {ar}\n#define KC {kc}\n" + KERNELS).get_function("analysis_warp")
                for ar, kc in ((2, 8), (4, 8), (8, 8))}
        for K in (8, 16, 24, 32, 40, 48, 64):
            fwd = cp.asarray(rng.normal(size=(2 * P, K)).astype(np.float16))
            fwd32 = fwd.astype(cp.float32)
            ref = field32 @ fwd32
            proj, projT = cp.zeros((2 * h, K), cp.float32), cp.zeros((K, 2 * h), cp.float32)

            def call(*args):
                status = gemm(gpu["handle"], *args)
                assert status == 0, f"cuBLAS status {status}"
            cases = {
                "analysis_cublas_fp16_transposed(production)": lambda: call(
                    1, 1, 2 * h, K, 2 * P, one.ctypes.data, field.data.ptr, 2, 2 * P, fwd.data.ptr, 2, K,
                    zero.ctypes.data, projT.data.ptr, 0, 2 * h, cb.CUBLAS_COMPUTE_32F, cb.CUBLAS_GEMM_DEFAULT),
                "analysis_cublas_fp16": lambda: call(
                    0, 0, K, 2 * h, 2 * P, one.ctypes.data, fwd.data.ptr, 2, K, field.data.ptr, 2, 2 * P,
                    zero.ctypes.data, proj.data.ptr, 0, K, cb.CUBLAS_COMPUTE_32F, cb.CUBLAS_GEMM_DEFAULT),
                "analysis_cublas_fp32": lambda: call(
                    0, 0, K, 2 * h, 2 * P, one.ctypes.data, fwd32.data.ptr, 0, K, field32.data.ptr, 0, 2 * P,
                    zero.ctypes.data, proj.data.ptr, 0, K, cb.CUBLAS_COMPUTE_32F, cb.CUBLAS_GEMM_DEFAULT),
                **{f"analysis_warp_kernel_AR{ar}_KC{kc}": (lambda ar=ar, kc=kc: warp[ar, kc](
                    ((2 * h // ar * 32 + 127) // 128,), (128,), (field, fwd32, proj, np.int32(2 * h), np.int32(2 * P), np.int32(K))))
                   for ar, kc in warp},
            }
            row = {}
            for name, f in cases.items():
                try:
                    proj[...] = 0
                    projT[...] = 0
                    f()
                    stream.synchronize()
                    got = projT.T if "transposed" in name else proj
                    err = float(cp.max(cp.abs(got - ref)) / cp.max(cp.abs(ref)))
                    us = graph_timer(cp, stream, f, reps=10, trials=5)[0]
                    row[name] = dict(us=us, rel_err=err, gflops=2 * 2 * h * 2 * P * K / (us * 1e-6) / 1e9)
                except Exception as error:  # noqa: BLE001
                    row[name] = dict(error=f"{type(error).__name__}: {error}")
            out[str(K)] = row
            log("gemm K=%d" % K, json.dumps({k: (round(v["us"], 1) if "us" in v else v["error"]) for k, v in row.items()}))
    return out


def section_kernels(cp, m, prepared, s, rounds):
    gpu = prepared["gpu"]
    out = {}
    for rname, (points, kept, dark) in rosters(m, prepared, s).items():
        if rname == "last":
            continue
        with gpu["stream"]:
            band, _ = bind(m, gpu, points, kept, dark)
            start_state(gpu, kept)
            m._rearrangement_roots(gpu, band)
            m._rearrangement_propagate(gpu, band, "encode")
            gpu["stream"].synchronize()
            launches = kernel_launches(m, gpu, band)
        times = {k: [] for k in launches}
        for r in range(rounds):
            for k, f in (list(launches.items()) if r % 2 == 0 else list(launches.items())[::-1]):
                times[k].append(graph_timer(cp, gpu["stream"], f, reps=10, trials=3)[0])
        summary = {k: dict(min_us=float(np.min(v)), median_us=float(np.median(v))) for k, v in times.items()}
        out[rname] = dict(points=len(points), kept=kept, dark=dark, band=band, timing=summary)
        log(f"kernels {s['name']} {rname}: points {len(points)} band {band}")
        for k, v in summary.items():
            log("   %-40s min %9.1f us   median %9.1f us" % (k, v["min_us"], v["median_us"]))
    return out


MAP_KINDS = {  # schedule kind -> roster role
    "release": "release", "crowded release": "release", "next": "next", "crowded next": "next",
    "first": "transport", "transport": "transport", "crowded transport": "transport", "many": "transport",
    "leap": "transport", "dark_last": "release", "crowded dark_last": "release",
    "last(filled)": "last", "last(refined, without its full-plane iterations)": "last"}


def section_maps(cp, m, prepared, s, rounds):
    """One map of every schedule kind, its captured graph(s) launched back to back on the scenario's real roster."""
    gpu = prepared["gpu"]
    roster = rosters(m, prepared, s)
    launches = {}
    for kind, role in MAP_KINDS.items():
        name = kind.split("(")[0]
        if name not in m._TRACK_SCHEDULE:
            continue
        points, kept, dark = roster[role]
        segments = m._rearrangement_segments(name, "refined" not in kind)
        with gpu["stream"]:
            band, row = bind(m, gpu, points, kept, dark, (m._TRACK_RELEASE_WINDOW if dark else m._TRACK_WINDOW,
                                                          m._TRACK_SCHEDULE[name][2], 1.1))
        graphs = [gpu["track_graphs"][(band, *segment)] for segment in segments]
        launches[kind] = (graphs, row, band, kept, sum(segment[0] + segment[1] + segment[2] for segment in segments))
    times = {k: [] for k in launches}
    for _ in range(rounds):
        for kind, (graphs, row, band, kept, evaluations) in launches.items():
            with gpu["stream"]:
                gpu["record"].set(row, stream=gpu["stream"])
                start_state(gpu, kept)

                def launch(graphs=graphs):
                    for graph in graphs:
                        graph.launch(gpu["stream"])
                times[kind].append(direct_timer(cp, gpu["stream"], launch, reps=3, trials=3)[0])
    out = {}
    for kind, (graphs, row, band, kept, evaluations) in launches.items():
        t = np.array(times[kind])
        out[kind] = dict(band=band, evaluations=evaluations, us=float(np.min(t)), median_us=float(np.median(t)),
                         per_evaluation_us=float(np.min(t)) / max(1, evaluations))
        log(f"maps {s['name']} {kind}", json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in out[kind].items()}))
    return out


def section_plane(cp, m, prepared, s):
    from cupyx.scipy import fft
    gpu = prepared["gpu"]
    points = np.asarray(s["target"])
    out = {}
    with gpu["stream"]:
        for iterations in sorted(set(m._TRACK_VACANT_PLANE)):
            def run(iterations=iterations):
                gpu["codes"][...] = gpu["target_codes"]
                m._rearrangement_plane_wgs(gpu, points, np.ones(len(points)), iterations, 0)
            out[f"plane_{iterations}_iterations_us"] = direct_timer(cp, gpu["stream"], run, reps=1, trials=3)[0]
        plane = cp.empty(SHAPE, cp.complex64)
        out["fft2_inplace_1024x1272_us"] = direct_timer(cp, gpu["stream"], lambda: fft.fft2(plane, overwrite_x=True), reps=5)[0]
        times = []
        for _ in range(5):
            gpu["stream"].synchronize()
            t0 = time.perf_counter()
            m._rearrangement_stray(gpu, points)
            times.append((time.perf_counter() - t0) * 1e3)
        out["last_map_stray_measurement_ms"] = float(np.median(times[1:]))
    log("plane", s["name"], json.dumps({k: round(v, 1) for k, v in out.items()}))
    return out


def section_ramp(cp, m, prepared, s):
    """A transport map launched 60 times right after 2 s of idle (heater running, as between shots), each launch
    timed on its own: the device time per launch against the time since the first launch."""
    gpu = prepared["gpu"]
    points, kept, dark = rosters(m, prepared, s)["transport"]
    segments = m._rearrangement_segments("transport", True)
    with active(m, prepared), gpu["stream"]:
        band, row = bind(m, gpu, points, kept, dark, (m._TRACK_WINDOW, m._TRACK_SCHEDULE["transport"][2], 1.1))
        graphs = [gpu["track_graphs"][(band, *segment)] for segment in segments]
        gpu["stream"].synchronize()
    out = {}
    for label in ("after_idle", "after_sustained_load"):
        if label == "after_idle":
            time.sleep(2.0)  # the heater runs here, as between shots
        with active(m, prepared), gpu["stream"]:
            if label == "after_sustained_load":
                start = time.perf_counter()
                while time.perf_counter() - start < 1.0:
                    for graph in graphs:
                        graph.launch(gpu["stream"])
                    gpu["stream"].synchronize()
            rows = []
            t_start = time.perf_counter()
            for _ in range(60):
                gpu["record"].set(row, stream=gpu["stream"])
                start_state(gpu, kept)
                a, b = cp.cuda.Event(), cp.cuda.Event()
                a.record(gpu["stream"])
                for graph in graphs:
                    graph.launch(gpu["stream"])
                b.record(gpu["stream"])
                b.synchronize()
                rows.append(((time.perf_counter() - t_start) * 1e3, cp.cuda.get_elapsed_time(a, b)))
        out[label] = rows
        device = np.array([r[1] for r in rows])
        log(f"ramp {label}: first 10 launches {np.round(device[:10], 2).tolist()} ms; last 10 median "
            f"{np.median(device[-10:]):.2f} ms; elapsed {rows[-1][0]:.0f} ms")
    return out


def section_seq(cp, mods, s, rounds):
    work = {}
    for v, m in mods.items():
        t0 = time.perf_counter()
        work[v] = prepare(m, s, v == "lpi")
        log(f"seq {s['name']} prepare {v}: {(time.perf_counter() - t0) * 1e3:.0f} ms")
    try:
        rows = {v: [] for v in mods}
        names = list(mods)
        for r in range(rounds + 1):
            for v in (names if r % 2 else names[::-1]):
                m, prepared = mods[v], work[v]
                stamps = []
                with contextlib.ExitStack() as stack:  # pause the other workspace's heater
                    for o in names:
                        if o != v:
                            stack.enter_context(active(mods[o], work[o]))
                    cp.cuda.Device().synchronize()
                    t0 = time.perf_counter()
                    plan = m.plan_rearrangement(prepared, s["occupied"])
                    result = m.compute_rearrangement(
                        prepared, plan, motion_frames=s["frames"], sampled=None, support_tolerance=1.01,
                        motion_support_tolerance=1.10, require_converged=False,
                        frame_ready=lambda i, f: stamps.append(time.perf_counter()))
                    t1 = time.perf_counter()
                if r:
                    rows[v].append(dict(photo_first_ms=(stamps[0] - t0) * 1e3, photo_last_ms=(stamps[-1] - t0) * 1e3,
                                        return_ms=(t1 - t0) * 1e3, maps=len(stamps),
                                        evaluations=int(np.sum(result.get("iterations", ()))),
                                        map_solve_ms=[float(x) for x in result.get("frame_solve_ms", ())],
                                        map_ready_ms=[float(x) for x in result.get("frame_ready_ms", ())]))
        out = dict(frames=s["frames"], rows=rows, summary={})
        for v, rs in rows.items():
            sm = {k: float(np.median([x[k] for x in rs])) for k in ("photo_first_ms", "photo_last_ms", "return_ms")}
            sm.update(maps=rs[0]["maps"], evaluations=rs[0]["evaluations"],
                      map_solve_ms_median=np.median([x["map_solve_ms"] for x in rs], axis=0).round(2).tolist()
                      if rs[0]["map_solve_ms"] else None)
            if v != "lpi":
                for k in ("photo_first_ms", "photo_last_ms", "return_ms"):
                    ratio = [a[k] / b[k] for a, b in zip(rs, rows["lpi"])]
                    sm[f"{k}_ratio_to_lpi"] = float(np.median(ratio))
                    sm[f"{k}_ratio_to_lpi_quartiles"] = [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))]
            out["summary"][v] = sm
            log(f"seq {s['name']} {v}", json.dumps(sm))
        return out
    finally:
        for prepared in work.values():
            prepared["close"]()


FRAME_MS = 1000 / 60
IDLE_S = 3.


def playback_end(ready_ms):
    """Photo -> end of a 60 Hz playback that shows each map once it is ready and its predecessor had its frame."""
    shown = None
    for ready in ready_ms:
        shown = ready if shown is None else max(ready, shown + FRAME_MS)
    return shown + FRAME_MS


def section_gap(cp, mods, s, rounds):
    """The lab's timing: the GPU idles IDLE_S (between shots and the exposure) with only the solver's own heater
    running, then the Fixed task path (plan + compute that samples itself), for this checkout and de73d721 (each with
    its own heater). 'warm': the same computation right after a sustained load (the bench's other sections).
    playback_end_ms simulates 60 Hz playback of the maps as they become ready."""
    work = {v: prepare(m, s, v == "lpi") for v, m in mods.items()}
    big = cp.zeros(16 << 20, cp.float32)
    load = cp.ElementwiseKernel("", "float32 x", "x += 1e-6f", "bench_busy")
    modes = [(v, mode) for v in mods for mode in ("warm", "idle")]
    rows = {f"{v} {mode}": [] for v, mode in modes}
    try:
        for r in range(rounds + 1):
            for v, mode in (modes if r % 2 else modes[::-1]):
                m, prepared = mods[v], work[v]
                stamps = []
                with contextlib.ExitStack() as stack:
                    for o in mods:  # one heater at a time, as in the product
                        if o != v:
                            stack.enter_context(active(mods[o], work[o]))
                    if mode == "warm":
                        with active(m, prepared):
                            start = time.perf_counter()
                            while time.perf_counter() - start < .3:
                                load(big)
                                cp.cuda.Device().synchronize()
                    else:
                        time.sleep(IDLE_S)  # only this workspace's heater runs
                    wall = time.strftime("%H:%M:%S")
                    t0 = time.perf_counter()
                    plan = m.plan_rearrangement(prepared, s["occupied"])
                    result = m.compute_rearrangement(
                        prepared, plan, motion_frames=s["frames"], sampled=None, support_tolerance=1.01,
                        motion_support_tolerance=1.10, require_converged=False,
                        frame_ready=lambda i, f: stamps.append(time.perf_counter()))
                    t1 = time.perf_counter()
                if r:
                    ready = [(x - t0) * 1e3 for x in stamps]
                    rows[f"{v} {mode}"].append(dict(
                        wall=wall, photo_first_ms=ready[0], photo_last_ms=ready[-1], playback_end_ms=playback_end(ready),
                        return_ms=(t1 - t0) * 1e3, map_solve_ms=[float(x) for x in result.get("frame_solve_ms", ())]))
        summary = {}
        for key, rs in rows.items():
            summary[key] = {k: float(np.median([x[k] for x in rs])) for k in ("photo_first_ms", "photo_last_ms",
                                                                            "playback_end_ms", "return_ms")}
            summary[key]["map_solve_ms_median"] = np.median([x["map_solve_ms"] for x in rs], axis=0).round(2).tolist()
        log(f"gap {s['name']} ({len(rs)} rounds, medians; playback = simulated 60 Hz end; nominal playback "
            f"{len(rows['new warm'][0]['map_solve_ms']) * FRAME_MS:.0f} ms)")
        for key, sm in summary.items():
            log("   %-18s photo->first %7.1f ms   photo->last %7.1f ms   photo->playback end %7.1f ms   maps %s" % (
                key, sm["photo_first_ms"], sm["photo_last_ms"], sm["playback_end_ms"],
                [round(x, 1) for x in sm["map_solve_ms_median"][:6]]))
        return dict(idle_s=IDLE_S, frames=s["frames"], rows=rows, summary=summary)
    finally:
        for prepared in work.values():
            prepared["close"]()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--sections", default="calib,gemm,kernels,maps,plane,ramp,seq,gap")
    args = ap.parse_args()
    sections = args.sections.split(",")
    stamp = f"slm_t400_{platform.node()}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(ROOT, "bench", "results", stamp)
    os.makedirs(out_dir)
    result = dict(host=platform.node(), python=sys.version, sections={}, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    result["git_head"] = git_head()
    smi = shutil.which("nvidia-smi")
    clocks = None
    if smi:
        with open(os.path.join(out_dir, "nvidia_smi.txt"), "w") as f:
            subprocess.run([smi, "-q"], stdout=f, stderr=subprocess.STDOUT)
        clocks = Clocks(smi, os.path.join(out_dir, "clocks.csv"))

    def save():
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=1, default=str)
        with open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(LOG) + "\n")

    def run(name, fn):
        if name.split(":")[0] not in sections:
            return
        log(f"=== {name}  {time.strftime('%Y/%m/%d %H:%M:%S')}")
        try:
            result["sections"][name] = fn()
        except Exception:  # noqa: BLE001
            result["sections"][name] = {"error": traceback.format_exc()}
            log(traceback.format_exc())
        save()

    try:
        try:
            import cupy as cp
        except ImportError:
            log("This Python has no CuPy: run bin\\install_slm_gpu.bat once, then this benchmark again.")
            raise
        mods = load_solvers()
        result.update(cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                      driver=cp.cuda.runtime.driverGetVersion(), solvers={k: m.__file__ for k, m in mods.items()})
        new = mods["new"]
        run("calib", lambda: section_calib(cp))
        for sname in ("real14", "real14c3"):
            if not any(x in sections for x in ("gemm", "kernels", "maps", "plane", "ramp")):
                break
            s = scenario(sname, new)
            prepared = prepare(new, s, False)
            try:
                with active(new, prepared):
                    if sname == "real14":
                        run("gemm", lambda: section_gemm(cp, new, prepared))
                        run(f"kernels:{sname}", lambda: section_kernels(cp, new, prepared, s, args.rounds))
                        run(f"plane:{sname}", lambda: section_plane(cp, new, prepared, s))
                    run(f"maps:{sname}", lambda: section_maps(cp, new, prepared, s, args.rounds))
                if sname == "real14":
                    run(f"ramp:{sname}", lambda: section_ramp(cp, new, prepared, s))
            finally:
                prepared["close"]()
        for sname in ("real14", "real14c3", "real14p13"):
            run(f"seq:{sname}", lambda sname=sname: section_seq(cp, mods, scenario(sname, new), args.rounds))
        for sname in ("real14", "real14c3", "real14p13"):
            run(f"gap:{sname}", lambda sname=sname: section_gap(cp, mods, scenario(sname, new), args.rounds))
    finally:
        if clocks is not None:
            clocks.close()
        result["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save()
        archive = shutil.make_archive(out_dir, "zip", out_dir)
        log("")
        log("Results folder:", out_dir)
        log("Send this file:", archive)
        save()


if __name__ == "__main__":
    main()
