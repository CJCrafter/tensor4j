package com.cjcrafter.tensor4j.memory;

/**
 * Device-agnostic memory allocator. Backed by a slab pool on CPU, by
 * {@code cudaMalloc} on CUDA, etc.
 */
public interface Allocator {

    /**
     * Allocates a block of at least {@code bytes}. The returned block must be
     * freed by invoking {@link DataPtr#deleter()}.
     */
    DataPtr allocate(long bytes);

    /**
     * Returns any unused slabs / blocks back to the OS / driver. Useful in
     * tests and on OOM.
     */
    void emptyCache();

    /**
     * Total bytes currently held by the allocator (live + cached).
     */
    long reservedBytes();

    /**
     * Total bytes currently handed out to live allocations.
     */
    long allocatedBytes();
}
