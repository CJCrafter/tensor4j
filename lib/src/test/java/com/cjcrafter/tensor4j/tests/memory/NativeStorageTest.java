package com.cjcrafter.tensor4j.tests.memory;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Device;
import com.cjcrafter.tensor4j.memory.CpuAllocator;
import com.cjcrafter.tensor4j.memory.NativeStorage;

import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NativeStorageTest {

    @Test
    void allocateProducesUsableF32Storage() {
        CpuAllocator a = new CpuAllocator();
        NativeStorage s = NativeStorage.allocate(a, DType.F32, 16);
        assertEquals(DType.F32, s.dtype());
        assertEquals(Device.cpu(), s.device());
        assertEquals(16 * Float.BYTES, s.sizeBytes());
        assertEquals(16, s.capacity());

        for (int i = 0; i < 16; i++)
            s.segment().setAtIndex(ValueLayout.JAVA_FLOAT, i, i);
        for (int i = 0; i < 16; i++)
            assertEquals((float) i, s.segment().getAtIndex(ValueLayout.JAVA_FLOAT, i));
    }

    @Test
    void deleteReturnsBlockToAllocator() {
        CpuAllocator a = new CpuAllocator();
        NativeStorage s = NativeStorage.allocate(a, DType.F32, 16);
        long beforeAllocated = a.allocatedBytes();
        assertTrue(beforeAllocated > 0);

        s.incRef();
        s.decRef();
        assertEquals(0, a.allocatedBytes());
    }

    @Test
    void refcountStartsAtZeroLikeOtherStorage() {
        NativeStorage s = NativeStorage.allocate(DType.F32, 8);
        assertEquals(0, s.useCount());
    }
}
