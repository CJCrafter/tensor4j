package com.cjcrafter.tensor4j.memory;

import com.google.common.base.Preconditions;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Block-split caching allocator over native shared arenas.
 *
 * <p>Allocations are rounded to a size class, satisfied from a pool of slabs,
 * and split into blocks. Freed blocks are coalesced with neighbors. Slabs are
 * only released back to the OS via {@link #emptyCache()}.
 */
public final class CpuAllocator implements Allocator {

    static final long SMALL_GRANULARITY = 512;
    static final long LARGE_GRANULARITY = 2L * 1024 * 1024;
    static final long SMALL_LIMIT = 1024L * 1024;
    static final long DEFAULT_SMALL_SLAB = 2L * 1024 * 1024;
    static final long DEFAULT_LARGE_SLAB = 20L * 1024 * 1024;
    static final long SIMD_ALIGN = 64;

    private static final CpuAllocator INSTANCE = new CpuAllocator();

    public static CpuAllocator instance() {
        return INSTANCE;
    }

    private final Object lock = new Object();
    private final List<Slab> smallSlabs = new ArrayList<>();
    private final List<Slab> largeSlabs = new ArrayList<>();

    public CpuAllocator() {
    }

    @Override
    public DataPtr allocate(long bytes) {
        Preconditions.checkArgument(bytes > 0, "bytes must be positive, got %s", bytes);
        long rounded = roundSize(bytes);
        boolean small = rounded <= SMALL_LIMIT;
        synchronized (lock) {
            List<Slab> pool = small ? smallSlabs : largeSlabs;
            for (Slab slab : pool) {
                Block block = slab.tryAllocate(rounded);
                if (block != null)
                    return wrap(slab, block);
            }
            long slabSize = Math.max(small ? DEFAULT_SMALL_SLAB : DEFAULT_LARGE_SLAB, rounded);
            Slab fresh = new Slab(slabSize);
            pool.add(fresh);
            Block block = fresh.tryAllocate(rounded);
            Preconditions.checkState(block != null, "fresh slab failed first allocation");
            return wrap(fresh, block);
        }
    }

    private DataPtr wrap(Slab slab, Block block) {
        MemorySegment view = slab.base.asSlice(block.offset, block.size);
        Runnable deleter = () -> {
            synchronized (lock) {
                slab.freeBlock(block);
            }
        };
        return new DataPtr(view, deleter);
    }

    @Override
    public void emptyCache() {
        synchronized (lock) {
            dropEmptySlabs(smallSlabs);
            dropEmptySlabs(largeSlabs);
        }
    }

    private static void dropEmptySlabs(List<Slab> pool) {
        Iterator<Slab> it = pool.iterator();
        while (it.hasNext()) {
            Slab slab = it.next();
            if (slab.liveBlocks == 0) {
                slab.close();
                it.remove();
            }
        }
    }

    @Override
    public long reservedBytes() {
        synchronized (lock) {
            long sum = 0;
            for (Slab s : smallSlabs) sum += s.size;
            for (Slab s : largeSlabs) sum += s.size;
            return sum;
        }
    }

    @Override
    public long allocatedBytes() {
        synchronized (lock) {
            long sum = 0;
            for (Slab s : smallSlabs) sum += s.allocatedBytes;
            for (Slab s : largeSlabs) sum += s.allocatedBytes;
            return sum;
        }
    }

    static long roundSize(long bytes) {
        long g = bytes <= SMALL_LIMIT ? SMALL_GRANULARITY : LARGE_GRANULARITY;
        return ((bytes + g - 1) / g) * g;
    }

    static final class Slab {
        final Arena arena;
        final MemorySegment base;
        final long size;
        int liveBlocks;
        long allocatedBytes;
        Block head;

        Slab(long size) {
            this.arena = Arena.ofShared();
            this.base = arena.allocate(size, SIMD_ALIGN);
            this.size = size;
            this.head = new Block(0, size, true);
        }

        Block tryAllocate(long bytes) {
            for (Block b = head; b != null; b = b.next) {
                if (b.free && b.size >= bytes) {
                    if (b.size > bytes) split(b, bytes);
                    b.free = false;
                    liveBlocks++;
                    allocatedBytes += b.size;
                    return b;
                }
            }
            return null;
        }

        private void split(Block b, long firstSize) {
            long remaining = b.size - firstSize;
            Block tail = new Block(b.offset + firstSize, remaining, true);
            tail.next = b.next;
            tail.prev = b;
            if (b.next != null) b.next.prev = tail;
            b.next = tail;
            b.size = firstSize;
        }

        void freeBlock(Block b) {
            Preconditions.checkState(!b.free, "block already free");
            b.free = true;
            liveBlocks--;
            allocatedBytes -= b.size;
            if (b.next != null && b.next.free) {
                Block n = b.next;
                b.size += n.size;
                b.next = n.next;
                if (n.next != null) n.next.prev = b;
            }
            if (b.prev != null && b.prev.free) {
                Block p = b.prev;
                p.size += b.size;
                p.next = b.next;
                if (b.next != null) b.next.prev = p;
            }
        }

        int blockCount() {
            int n = 0;
            for (Block b = head; b != null; b = b.next) n++;
            return n;
        }

        void close() {
            arena.close();
        }
    }

    static final class Block {
        long offset;
        long size;
        boolean free;
        Block prev;
        Block next;

        Block(long offset, long size, boolean free) {
            this.offset = offset;
            this.size = size;
            this.free = free;
        }
    }
}
