package com.cjcrafter.tensor4j.memory;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Device;

/**
 * A contiguous block of memory that stores the underlying data of a tensor
 * on some device, backed by a plain heap array. Lifecycle is managed by the
 * JVM garbage collector; no manual refcounting is needed.
 */
public abstract class Storage {

    public abstract DType dtype();

    public abstract Device device();

    public abstract long sizeBytes();

    public final long capacity() {
        return sizeBytes() / dtype().sizeBytes();
    }

    /**
     * Returns the backing {@code float[]} for this storage.
     */
    public abstract float[] array();
}
