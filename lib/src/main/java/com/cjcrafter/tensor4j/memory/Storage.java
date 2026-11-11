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
     *
     * @return the element type
     */
    public abstract DType dtype();

    /**
     * Returns the device this block's memory lives on.
     *
     * @return the device
     */
    public abstract Device device();

    /**
     * Returns the total size of this block in bytes.
     *
     * @return the computed value
     */
    public abstract long sizeBytes();

    /**
     * Returns the number of elements this block can hold ({@code sizeBytes() / dtype().sizeBytes()}).
      *
     * @return the computed value
     */
    public final long capacity() {
        return sizeBytes() / dtype().sizeBytes();
    }

    /**
     * Returns the backing {@code float[]} for this storage.
      *
     * @return a copy of the values, in row-major order
     */
    public abstract float[] array();
}
