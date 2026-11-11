package com.cjcrafter.tensor4j.backend;

import com.cjcrafter.tensor4j.Device;
import com.cjcrafter.tensor4j.memory.Allocator;

/**
 * Per-{@link Device} compute facade. Owns the device's {@link Allocator} and
 * exposes per-{@code DType} kernel tables. Resolved via {@link Backends}.
 */
public interface Backend {

    /**
     * The device this backend computes on.
     */
    Device device();

    /**
     * The allocator that produces {@code Storage} blocks on this device.
     */
    Allocator allocator();

    /**
     * Returns the F32 kernel table for this backend.
     */
    F32KernelTable f32Kernels();
}
