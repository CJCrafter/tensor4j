package com.cjcrafter.tensor4j.backend.cpu;

import com.cjcrafter.tensor4j.Device;
import com.cjcrafter.tensor4j.backend.Backend;
import com.cjcrafter.tensor4j.backend.F32KernelTable;
import com.cjcrafter.tensor4j.memory.Allocator;
import com.cjcrafter.tensor4j.memory.CpuAllocator;

/**
 * CPU compute backend. Uses {@link CpuAllocator#instance()} for allocation.
 */
public final class CpuBackend implements Backend {

    private static final CpuBackend INSTANCE = new CpuBackend();

    public static CpuBackend instance() {
        return INSTANCE;
    }

    private CpuBackend() {}

    @Override
    public Device device() {
        return Device.cpu();
    }

    @Override
    public Allocator allocator() {
        return CpuAllocator.instance();
    }

    @Override
    public F32KernelTable f32Kernels() {
        return CpuF32Kernels.instance();
    }
}
