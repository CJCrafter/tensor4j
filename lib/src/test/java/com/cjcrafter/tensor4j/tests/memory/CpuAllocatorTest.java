package com.cjcrafter.tensor4j.tests.memory;

import com.cjcrafter.tensor4j.memory.CpuAllocator;
import com.cjcrafter.tensor4j.memory.DataPtr;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CpuAllocatorTest {

    @Test
    void allocateAndFreeRoundTrips() {
        CpuAllocator a = new CpuAllocator();
        DataPtr p = a.allocate(1024);
        assertNotNull(p.segment());
        assertTrue(p.byteSize() >= 1024);
        assertEquals(p.byteSize(), a.allocatedBytes());
        p.deleter().run();
        assertEquals(0, a.allocatedBytes());
    }

    @Test
    void segmentIsWritableAndReadable() {
        CpuAllocator a = new CpuAllocator();
        DataPtr p = a.allocate(4 * 64);
        MemorySegment seg = p.segment();
        for (int i = 0; i < 64; i++)
            seg.setAtIndex(ValueLayout.JAVA_FLOAT, i, i * 1.5f);
        for (int i = 0; i < 64; i++)
            assertEquals(i * 1.5f, seg.getAtIndex(ValueLayout.JAVA_FLOAT, i));
        p.deleter().run();
    }

    @Test
    void multipleSmallAllocationsShareOneSlab() {
        CpuAllocator a = new CpuAllocator();
        DataPtr p1 = a.allocate(1024);
        long reservedAfterFirst = a.reservedBytes();
        DataPtr p2 = a.allocate(1024);
        DataPtr p3 = a.allocate(1024);
        assertEquals(reservedAfterFirst, a.reservedBytes(),
                "additional small allocs should fit in the first slab");
        assertNotEquals(p1.segment().address(), p2.segment().address());
        assertNotEquals(p2.segment().address(), p3.segment().address());
        p1.deleter().run();
        p2.deleter().run();
        p3.deleter().run();
    }

    @Test
    void freedBlocksAreReusedAfterCoalescing() {
        CpuAllocator a = new CpuAllocator();
        DataPtr p1 = a.allocate(1024);
        DataPtr p2 = a.allocate(1024);
        long reserved = a.reservedBytes();
        p1.deleter().run();
        p2.deleter().run();
        DataPtr p3 = a.allocate(1024);
        assertEquals(reserved, a.reservedBytes(),
                "reuse must not grow reservedBytes");
        p3.deleter().run();
    }

    @Test
    void emptyCacheReleasesUnusedSlabs() {
        CpuAllocator a = new CpuAllocator();
        DataPtr p = a.allocate(1024);
        assertTrue(a.reservedBytes() > 0);
        p.deleter().run();
        a.emptyCache();
        assertEquals(0, a.reservedBytes());
        assertEquals(0, a.allocatedBytes());
    }

    @Test
    void emptyCacheKeepsSlabsWithLiveBlocks() {
        CpuAllocator a = new CpuAllocator();
        DataPtr live = a.allocate(1024);
        long reserved = a.reservedBytes();
        a.emptyCache();
        assertEquals(reserved, a.reservedBytes(),
                "slabs with live blocks must not be released");
        live.deleter().run();
        a.emptyCache();
        assertEquals(0, a.reservedBytes());
    }

    @Test
    void smallAndLargeUseDifferentPools() {
        CpuAllocator a = new CpuAllocator();
        DataPtr small = a.allocate(1024);
        DataPtr large = a.allocate(8L * 1024 * 1024);
        assertTrue(a.reservedBytes() >= small.byteSize() + large.byteSize());
        small.deleter().run();
        large.deleter().run();
        a.emptyCache();
    }

    @Test
    void allocationsRoundUpToGranularity() {
        CpuAllocator a = new CpuAllocator();
        DataPtr p = a.allocate(1);
        assertTrue(p.byteSize() >= 512,
                "small request must round up to granularity");
        p.deleter().run();
        a.emptyCache();
    }

    @Test
    void hugeRequestCreatesOversizedSlab() {
        CpuAllocator a = new CpuAllocator();
        long huge = 64L * 1024 * 1024;
        DataPtr p = a.allocate(huge);
        assertTrue(a.reservedBytes() >= huge);
        p.deleter().run();
        a.emptyCache();
    }

    @Test
    void doubleFreeThrows() {
        CpuAllocator a = new CpuAllocator();
        DataPtr p = a.allocate(1024);
        p.deleter().run();
        assertThrows(IllegalStateException.class, () -> p.deleter().run());
    }
}
