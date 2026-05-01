package com.cjcrafter.tensor4j.memory;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Device;

import java.util.concurrent.atomic.AtomicLong;

/**
 * A contiguous block of memory that stores the underlying
 * data of a tensor on some device. Identity-bearing and refcounted.
 */
public abstract class Storage {

    private final AtomicLong refCount = new AtomicLong(0);

    public abstract DType dtype();

    public abstract Device device();

    public abstract long sizeBytes();

    public final long capacity() {
        return sizeBytes() / dtype().sizeBytes();
    }

    public final long incRef() {
        return refCount.incrementAndGet();
    }

    public final long decRef() {
        long n = refCount.decrementAndGet();
        if (n == 0)
            delete();
        else if (n < 0)
            throw new IllegalStateException("decRef below zero on " + this);
        return n;
    }

    public final long useCount() {
        return refCount.get();
    }

    /**
     * Releases the underlying buffer. Invoked exactly once when the strong refcount hits zero.
     */
    protected abstract void delete();
}
