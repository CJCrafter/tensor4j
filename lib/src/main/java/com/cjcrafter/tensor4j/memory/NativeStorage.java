package com.cjcrafter.tensor4j.memory;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Device;
import com.google.common.base.Preconditions;

import java.lang.foreign.MemorySegment;

/**
 * CPU storage backed by a {@link MemorySegment} carved from a {@link CpuAllocator}.
 * On {@link #delete()} the block is returned to its allocator's pool.
 */
public final class NativeStorage extends Storage {

    private final DataPtr ptr;
    private final DType dtype;
    private final long sizeBytes;
    private final MemorySegment view;

    public NativeStorage(DataPtr ptr, DType dtype, long sizeBytes) {
        this.ptr = Preconditions.checkNotNull(ptr, "ptr");
        this.dtype = Preconditions.checkNotNull(dtype, "dtype");
        Preconditions.checkArgument(sizeBytes > 0 && sizeBytes <= ptr.byteSize(),
                "sizeBytes %s exceeds block %s", sizeBytes, ptr.byteSize());
        this.sizeBytes = sizeBytes;
        this.view = ptr.segment().asSlice(0, sizeBytes);
    }

    public static NativeStorage allocate(DType dtype, long numElements) {
        return allocate(CpuAllocator.instance(), dtype, numElements);
    }

    public static NativeStorage allocate(Allocator allocator, DType dtype, long numElements) {
        Preconditions.checkNotNull(allocator, "allocator");
        Preconditions.checkNotNull(dtype, "dtype");
        Preconditions.checkArgument(numElements > 0,
                "numElements must be positive, got %s", numElements);
        long bytes = numElements * dtype.sizeBytes();
        return new NativeStorage(allocator.allocate(bytes), dtype, bytes);
    }

    @Override
    public MemorySegment segment() {
        return view;
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
        return sizeBytes;
    }

    @Override
    protected void delete() {
        ptr.deleter().run();
    }
}
