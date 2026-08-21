"""
Model Manager - Universal Polymorphic Resource Governor for Apple Silicon Unified Memory.

Responsibilities:
- Polymorphic registry for all model pipelines (LLM, Image, Video, Audio)
- Track unified memory usage across MLX (Metal) and PyTorch (MPS) frameworks
- Limit per-pipeline concurrency via dynamic asyncio.Semaphore instances
- Run blocking inference in a dedicated ThreadPoolExecutor
- Evict idle pipelines (LRU) when memory pressure exceeds the 110GB budget
- Provide real-time system & pipeline telemetry via /v1/system/status
"""

import asyncio
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type
from functools import partial

from aistudio.utils.logging import logger
from aistudio.pipelines.base import BasePipeline


@dataclass
class PipelineState:
    """Tracks runtime state for a registered pipeline."""
    pipeline_type: str
    last_used: float = field(default_factory=time.time)
    active_requests: int = 0


class ModelManager:
    """
    Central unified resource governor for Apple Silicon Unified Memory architecture.

    Features:
    - Unified Registry: Pipelines register via `register_pipeline(pipeline)`.
    - Dynamic Concurrency: Dynamically creates and manages FIFO semaphores per pipeline type.
    - Non-blocking Execution: Dispatches synchronous inference steps onto worker threads.
    - Memory Management: 110GB memory budget with LRU pipeline eviction.
    """

    def __init__(self, memory_budget_gb: float = 110.0, max_workers: int = 4):
        self.memory_budget_bytes = int(memory_budget_gb * 1024**3)

        # Polymorphic pipeline registry: {pipeline_type: BasePipeline instance}
        self._pipelines: Dict[str, BasePipeline] = {}

        # Per-pipeline concurrency limiters (dynamically expanded upon registration)
        self._semaphores: Dict[str, asyncio.Semaphore] = {}

        # Pipeline runtime state tracking
        self._pipeline_states: Dict[str, PipelineState] = {}

        # Thread pool for non-blocking inference
        self._thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="model-worker"
        )

        # Async lock for state mutations
        self._state_lock = asyncio.Lock()

        # Telemetry metrics
        self._requests_queued = 0
        self._requests_completed = 0
        self._requests_timed_out = 0
        self._start_time = time.time()

        # Auto-register core pipelines
        self._auto_register_pipelines()

        logger.info(
            f"⚡ Universal ModelManager initialized: "
            f"{memory_budget_gb:.0f}GB memory budget, "
            f"{max_workers} worker threads, "
            f"{len(self._pipelines)} pipelines registered"
        )

    # ── Pipeline Registry ───────────────────────────────────────────────────

    def register_pipeline(self, pipeline: BasePipeline) -> None:
        """
        Registers a pipeline into the universal model manager.
        Automatically provisions concurrency semaphores and state tracking.
        """
        ptype = pipeline.pipeline_type
        self._pipelines[ptype] = pipeline

        if ptype not in self._semaphores:
            self._semaphores[ptype] = asyncio.Semaphore(1)

        if ptype not in self._pipeline_states:
            self._pipeline_states[ptype] = PipelineState(pipeline_type=ptype)

        logger.info(f"Registered pipeline '{ptype}' into universal ModelManager")

    def get_pipeline(self, pipeline_type: str) -> Optional[BasePipeline]:
        """Retrieve a registered pipeline by its modality type."""
        return self._pipelines.get(pipeline_type)

    def list_pipelines(self) -> Dict[str, Any]:
        """Returns runtime status for all registered pipelines."""
        result = {}
        for ptype, pipe in self._pipelines.items():
            info = pipe.get_info()
            state = self._pipeline_states.get(ptype)
            info["active_requests"] = state.active_requests if state else 0
            info["last_used_ago_seconds"] = round(time.time() - state.last_used, 1) if state else 0
            result[ptype] = info
        return result

    def _auto_register_pipelines(self):
        """Auto-discovers and registers standard pipelines."""
        try:
            from aistudio.pipelines.llm import llm_pipeline
            self.register_pipeline(llm_pipeline)
        except Exception as e:
            logger.warning(f"Could not auto-register LLM pipeline: {e}")

        try:
            from aistudio.pipelines.image import image_pipeline
            self.register_pipeline(image_pipeline)
        except Exception as e:
            logger.warning(f"Could not auto-register Image pipeline: {e}")

        try:
            from aistudio.pipelines.video import video_pipeline
            self.register_pipeline(video_pipeline)
        except Exception as e:
            logger.warning(f"Could not auto-register Video pipeline: {e}")

        try:
            from aistudio.pipelines.audio import audio_pipeline
            self.register_pipeline(audio_pipeline)
        except Exception as e:
            logger.warning(f"Could not auto-register Audio pipeline: {e}")

    # ── Concurrency & Resource Acquisition ──────────────────────────────────

    @asynccontextmanager
    async def acquire(self, pipeline_type: str):
        """
        Async context manager that governs access to a pipeline:
        1. Ensures total unified memory is within the 110GB budget (evicts LRU if needed)
        2. Acquires the per-pipeline FIFO semaphore
        3. Tracks active requests
        4. Releases slots on completion or exception
        """
        self._requests_queued += 1

        # Ensure semaphore exists even if registered dynamically
        if pipeline_type not in self._semaphores:
            self._semaphores[pipeline_type] = asyncio.Semaphore(1)

        # Phase 1: Wait for memory budget
        await self._ensure_memory_available(pipeline_type)

        # Phase 2: Acquire pipeline slot (FIFO queue)
        await self._semaphores[pipeline_type].acquire()

        try:
            # Track active request
            async with self._state_lock:
                if pipeline_type not in self._pipeline_states:
                    self._pipeline_states[pipeline_type] = PipelineState(
                        pipeline_type=pipeline_type
                    )
                state = self._pipeline_states[pipeline_type]
                state.active_requests += 1
                state.last_used = time.time()

            yield

        finally:
            # Always release on exit
            async with self._state_lock:
                if pipeline_type in self._pipeline_states:
                    self._pipeline_states[pipeline_type].active_requests -= 1

            self._semaphores[pipeline_type].release()
            self._requests_completed += 1

    async def run_in_thread(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Run a blocking inference function in the thread pool so it
        doesn't freeze FastAPI's asyncio event loop.
        """
        loop = asyncio.get_running_loop()
        func = partial(fn, *args, **kwargs)
        return await loop.run_in_executor(self._thread_pool, func)

    # ── Memory & System Telemetry ───────────────────────────────────────────

    def get_memory_usage_bytes(self) -> int:
        """
        Get current ML framework memory usage in bytes.
        On Apple Silicon, MLX and PyTorch MPS share unified memory.
        """
        total = 0

        try:
            import mlx.core as mx
            if hasattr(mx, "get_active_memory"):
                total += mx.get_active_memory()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                total += mx.metal.get_active_memory()
        except Exception:
            pass

        try:
            import torch
            if torch.backends.mps.is_available():
                total += torch.mps.current_allocated_memory()
        except Exception:
            pass

        return total

    def get_system_stats(self) -> Dict[str, Any]:
        """Return comprehensive system resource statistics."""
        mlx_mem = 0
        mlx_peak = 0
        mps_mem = 0

        try:
            import mlx.core as mx
            if hasattr(mx, "get_active_memory"):
                mlx_mem = mx.get_active_memory()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                mlx_mem = mx.metal.get_active_memory()

            if hasattr(mx, "get_peak_memory"):
                mlx_peak = mx.get_peak_memory()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
                mlx_peak = mx.metal.get_peak_memory()
        except Exception:
            pass

        try:
            import torch
            if torch.backends.mps.is_available():
                mps_mem = torch.mps.current_allocated_memory()
        except Exception:
            pass

        system_mem = {}
        cpu_info = {}
        try:
            import psutil
            vm = psutil.virtual_memory()
            system_mem = {
                "total_gb": round(vm.total / 1e9, 1),
                "available_gb": round(vm.available / 1e9, 1),
                "used_gb": round(vm.used / 1e9, 1),
                "percent": vm.percent,
            }
            cpu_info = {
                "cores_physical": psutil.cpu_count(logical=False),
                "cores_logical": psutil.cpu_count(logical=True),
                "percent": psutil.cpu_percent(interval=0.1),
            }
        except Exception:
            pass

        # Pipeline statuses
        pipelines = self.list_pipelines()

        # Queue depth per semaphore
        queue_depths = {}
        for ptype, sem in self._semaphores.items():
            waiters = getattr(sem, "_waiters", None)
            waiting = len(waiters) if waiters is not None else 0
            queue_depths[ptype] = {
                "available_slots": sem._value,
                "waiting_requests": waiting,
            }

        uptime = time.time() - self._start_time
        total_ml_bytes = mlx_mem + mps_mem

        return {
            "uptime_seconds": round(uptime, 1),
            "memory": {
                "budget_gb": round(self.memory_budget_bytes / 1e9, 1),
                "mlx_active_gb": round(mlx_mem / 1e9, 2),
                "mlx_peak_gb": round(mlx_peak / 1e9, 2),
                "mps_allocated_gb": round(mps_mem / 1e9, 2),
                "total_ml_gb": round(total_ml_bytes / 1e9, 2),
                "budget_used_percent": round(
                    total_ml_bytes / max(1, self.memory_budget_bytes) * 100, 1
                ),
                "system": system_mem,
            },
            "cpu": cpu_info,
            "pipelines": pipelines,
            "queues": queue_depths,
            "throughput": {
                "total_queued": self._requests_queued,
                "total_completed": self._requests_completed,
                "total_timed_out": self._requests_timed_out,
            },
            "thread_pool": {
                "max_workers": self._thread_pool._max_workers,
            },
        }

    # ── Internal Eviction & Memory Handling ─────────────────────────────────

    async def _ensure_memory_available(self, pipeline_type: str):
        """
        If memory usage exceeds budget, attempt to evict idle pipelines.
        Proceed immediately once idle pipelines have been evicted.
        """
        current = self.get_memory_usage_bytes()
        if current >= self.memory_budget_bytes:
            await self._evict_idle_pipeline(exclude=pipeline_type)
        return

    async def _evict_idle_pipeline(self, exclude: str = None) -> bool:
        """
        Unload the least recently used pipeline that has no active requests.
        Returns True if something was freed.
        """
        async with self._state_lock:
            candidates = [
                (ptype, state)
                for ptype, state in self._pipeline_states.items()
                if ptype != exclude and state.active_requests == 0
            ]

            if not candidates:
                return False

            # Evict least recently used
            candidates.sort(key=lambda x: x[1].last_used)
            ptype, _ = candidates[0]

            logger.info(f"🗑️  Evicting idle pipeline '{ptype}' to free memory (LRU)")
            self._unload_pipeline(ptype)
            return True

    def _unload_pipeline(self, pipeline_type: str):
        """Synchronously unload a pipeline and free all associated memory."""
        pipeline = self._pipelines.get(pipeline_type)
        if pipeline:
            try:
                pipeline.unload()
            except Exception as e:
                logger.error(f"Error unloading '{pipeline_type}': {e}")

        gc.collect()

        # Clear ML framework caches
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass

        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        logger.info(f"✅ Pipeline '{pipeline_type}' unloaded polymorphically, caches cleared")


# ── Singleton ───────────────────────────────────────────────────────────────
model_manager = ModelManager()
