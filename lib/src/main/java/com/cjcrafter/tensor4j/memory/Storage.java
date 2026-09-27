package com.cjcrafter.tensor4j.memory;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Device;

/**
 * A contiguous block of memory that stores the underlying data of a tensor
 * on some device, backed by a plain heap array. Lifecycle is managed by the
 * JVM garbage collector; no manual refcounting is needed.
 */
public abstract class Storage {

    /**
     * Returns the element type stored in this block.
     */
    public abstract DType dtype();

    /**
     * Returns the device this block's memory lives on.
     */
    public abstract Device device();

    /**
     * Returns the total size of this block in bytes.
     */
    public abstract long sizeBytes();

    /**
     * Returns the number of elements this block can hold ({@code sizeBytes() / dtype().sizeBytes()}).
     */
    public final long capacity() {
        return sizeBytes() / dtype().sizeBytes();
    }

    /**
     * Returns the backing {@code float[]} for this storage.
     */
    public abstract float[] array();
}
