package com.cjcrafter.tensor4j.memory;

import com.google.common.base.Preconditions;

import java.lang.foreign.MemorySegment;

/**
 * A handle to a chunk of memory carved out of an {@link Allocator}. Pairs a
 * {@link MemorySegment} view with a {@link Runnable} that returns the block
 * to its allocator.
 */
public final class DataPtr {

    private final MemorySegment segment;
    private final Runnable deleter;

    public DataPtr(MemorySegment segment, Runnable deleter) {
        this.segment = Preconditions.checkNotNull(segment, "segment");
        this.deleter = Preconditions.checkNotNull(deleter, "deleter");
    }

    public MemorySegment segment() {
        return segment;
    }

    public Runnable deleter() {
        return deleter;
    }

    public long byteSize() {
        return segment.byteSize();
    }
}
