package com.cjcrafter.tensor4j.memory;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Device;
import com.google.common.base.Preconditions;

/**
 * CPU storage backed by a plain {@code float[]} on the Java heap.
 */
public final class HeapStorage extends Storage {

    private final float[] data;
    private final DType dtype;

    /**
     * Wraps an existing array as storage; the array is used directly, not copied.
     *
     * @param data the values, in row-major order
     * @param dtype the element type
     */
    public HeapStorage(float[] data, DType dtype) {
        this.data = Preconditions.checkNotNull(data, "data");
        this.dtype = Preconditions.checkNotNull(dtype, "dtype");
    }

    /**
     * Allocates a new zero-filled storage block of {@code numElements} elements.
     * @param dtype the element type
     * @param numElements the number of elements to allocate
     * @return a new storage block
     */
    public static HeapStorage allocate(DType dtype, int numElements) {
        Preconditions.checkNotNull(dtype, "dtype");
        Preconditions.checkArgument(numElements > 0,
                "numElements must be positive, got %s", numElements);
        return new HeapStorage(new float[numElements], dtype);
    }

    @Override
    public float[] array() {
        return data;
    }

    @Override
    public DType dtype() {
        return dtype;
    }

    @Override
    public Device device() {
        return Device.cpu();
    }

    @Override
    public long sizeBytes() {
        return (long) data.length * dtype.sizeBytes();
    }
}
