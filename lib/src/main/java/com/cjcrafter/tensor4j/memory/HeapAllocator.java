package com.cjcrafter.tensor4j.memory;

import com.cjcrafter.tensor4j.DType;

/**
 * Allocates plain heap-backed {@link HeapStorage}; the JVM garbage collector
 * owns the lifecycle, so there is no pooling or manual release.
 */
public final class HeapAllocator implements Allocator {

    private static final HeapAllocator INSTANCE = new HeapAllocator();

    /**
     * Returns the shared singleton instance.
     */
    public static HeapAllocator instance() {
        return INSTANCE;
    }

    /**
     * Creates a new allocator instance; since heap allocation has no shared pool or state,
     * this behaves identically to {@link #instance()}.
     */
    public HeapAllocator() {
    }

    @Override
    public Storage allocate(DType dtype, int numElements) {
        return HeapStorage.allocate(dtype, numElements);
    }
}
