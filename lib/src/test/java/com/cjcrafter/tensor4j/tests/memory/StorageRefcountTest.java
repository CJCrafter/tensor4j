package com.cjcrafter.tensor4j.tests.memory;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Device;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.memory.NativeStorage;
import com.cjcrafter.tensor4j.memory.Storage;

import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class StorageRefcountTest {

    @Test
    void freshStorageStartsAtZero() {
        Storage s = NativeStorage.allocate(DType.F32, 16);
        assertEquals(0, s.useCount());
    }

    @Test
    void incDecRoundTrips() {
        Storage s = NativeStorage.allocate(DType.F32, 16);
        s.incRef();
        s.incRef();
        assertEquals(2, s.useCount());
        s.decRef();
        assertEquals(1, s.useCount());
        s.decRef();
        assertEquals(0, s.useCount());
    }

    @Test
    void decRefBelowZeroThrows() {
        Storage s = NativeStorage.allocate(DType.F32, 16);
        assertThrows(IllegalStateException.class, s::decRef);
    }

    private static final class CountingStorage extends Storage {
        final AtomicInteger deletes = new AtomicInteger(0);
        @Override public DType dtype() { return DType.F32; }
        @Override public Device device() { return Device.cpu(); }
        @Override public long sizeBytes() { return 16; }
        @Override public MemorySegment segment() { return MemorySegment.NULL; }
        @Override protected void delete() { deletes.incrementAndGet(); }
    }

    @Test
    void deleteFiresExactlyOnceAtZero() {
        CountingStorage s = new CountingStorage();
        s.incRef();
        s.incRef();
        s.decRef();
        assertEquals(0, s.deletes.get());
        s.decRef();
        assertEquals(1, s.deletes.get());
    }

    @Test
    void freshTensorHasRefcountOne() {
        Tensor t = Tensor4j.builder().shape(4, 4).fill(1f);
        assertEquals(1, t.getStorage().useCount());
    }

    @Test
    void viewsShareStorageRefcount() {
        Tensor t = Tensor4j.builder().shape(4, 4).fill(1f);
        Storage s = t.getStorage();
        assertEquals(1, s.useCount());

        Tensor v = t.view(2, 8);
        assertEquals(2, s.useCount());

        Tensor tr = t.transpose();
        assertEquals(3, s.useCount());

        Tensor d = t.detach();
        assertEquals(4, s.useCount());

        assertNotNull(v);
        assertNotNull(tr);
        assertNotNull(d);
    }

    @Test
    void cleanerEventuallyDecrefsStorage() throws InterruptedException {
        Tensor t = Tensor4j.builder().shape(4, 4).fill(1f);
        Storage s = t.getStorage();
        assertEquals(1, s.useCount());

        t = null;
        for (int i = 0; i < 50 && s.useCount() > 0; i++) {
            System.gc();
            Thread.sleep(20);
        }
        assertEquals(0, s.useCount());
    }

    @Test
    void tanhDetachShareStorageWithResult() {
        Tensor x = Tensor4j.builder().shape(2, 2).requiresGrad().rand();
        Tensor y = x.tanh();
        Storage ys = y.getStorage();
        // y holds storage (ref 1). TanhFn saves y.detach() which also holds (ref 2).
        assertEquals(2, ys.useCount());
    }
}
