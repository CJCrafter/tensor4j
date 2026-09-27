package com.cjcrafter.tensor4j.memory;

import com.cjcrafter.tensor4j.DType;

/**
 * Allocates plain heap-backed {@link HeapStorage}; the JVM garbage collector
 * owns the lifecycle, so there is no pooling or manual release.
 */
public final class HeapAllocator implements Allocator {

    private static final HeapAllocator INSTANCE = new HeapAllocator();

    public static HeapAllocator instance() {
        return INSTANCE;
    }

    public HeapAllocator() {
    }

    @Override
    public Storage allocate(DType dtype, int numElements) {
        return HeapStorage.allocate(dtype, numElements);
    }
}
