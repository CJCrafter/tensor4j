package com.cjcrafter.tensor4j.tests.memory;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Device;
import com.cjcrafter.tensor4j.memory.HeapAllocator;
import com.cjcrafter.tensor4j.memory.HeapStorage;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class HeapStorageTest {

    @Test
    void allocateProducesUsableF32Storage() {
        HeapStorage s = HeapStorage.allocate(DType.F32, 16);
        assertEquals(DType.F32, s.dtype());
        assertEquals(Device.cpu(), s.device());
        assertEquals(16L * Float.BYTES, s.sizeBytes());
        assertEquals(16, s.capacity());

        float[] arr = s.array();
        for (int i = 0; i < 16; i++)
            arr[i] = i;
        for (int i = 0; i < 16; i++)
            assertEquals((float) i, s.array()[i]);
    }

    @Test
    void allocateRejectsNonPositiveSize() {
        assertThrows(IllegalArgumentException.class, () -> HeapStorage.allocate(DType.F32, 0));
    }

    @Test
    void heapAllocatorAllocatesBackingArrayOfRequestedSize() {
        HeapAllocator a = new HeapAllocator();
        var s = a.allocate(DType.F32, 8);
        assertEquals(8, s.array().length);
        assertEquals(Device.cpu(), s.device());
    }
}
