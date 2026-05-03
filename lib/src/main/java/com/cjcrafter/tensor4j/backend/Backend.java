package com.cjcrafter.tensor4j.backend;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Device;
import com.cjcrafter.tensor4j.F32Type;
import com.cjcrafter.tensor4j.memory.Allocator;

/**
 * Per-{@link Device} compute facade. Owns the device's {@link Allocator} and
 * exposes per-{@link DType} kernel tables. Resolved via {@link Backends}.
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
     * Returns the F32 kernel table for this backend, or throws if F32 is not
     * supported on this device.
     *
     * @throws UnsupportedOperationException If F32 is unsupported.
     */
    F32KernelTable f32Kernels();

    /**
     * Returns whether this backend can compute on tensors of the given dtype.
     * Useful for fast probes before constructing a tensor.
     */
    default boolean supports(DType dtype) {
        // TODO: support more than floats
        return dtype instanceof F32Type;
    }
}
