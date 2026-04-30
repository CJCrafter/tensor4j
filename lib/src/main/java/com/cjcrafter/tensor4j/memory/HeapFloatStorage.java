package com.cjcrafter.tensor4j.memory;

import com.cjcrafter.tensor4j.DType;
import com.google.common.base.Preconditions;

/**
 * Heap-backed F32 storage wrapping a {@code float[]}.
 */
public final class HeapFloatStorage extends HeapStorage {

    private final float[] array;

    public HeapFloatStorage(float[] array) {
        Preconditions.checkNotNull(array, "array");
        Preconditions.checkArgument(array.length > 0, "storage must be non-empty");
        this.array = array;
    }

    public static HeapFloatStorage allocate(int numElements) {
        Preconditions.checkArgument(numElements > 0,
                "numElements must be positive, got %s", numElements);
        return new HeapFloatStorage(new float[numElements]);
    }

    /**
     * Direct access to the backing array. Not defensively copied.
     */
    public float[] array() {
        return array;
    }

    @Override
    public DType dtype() {
        return DType.F32;
    }

    @Override
    public long sizeBytes() {
        return (long) array.length * Float.BYTES;
    }
}
