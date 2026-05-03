package com.cjcrafter.tensor4j.tests.memory;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.memory.CpuAllocator;
import com.cjcrafter.tensor4j.memory.NativeStorage;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Exercises the full pipeline: TensorBuilder allocator() -> NativeStorage tensors,
 * dispatched through Ops -> SegmentKernels, and verifies parity with the heap path.
 */
class NativeTensorE2ETest {

    private static final float EPS = 1e-5f;

    @Test
    void builderProducesNativeStorageWhenAllocatorSet() {
        CpuAllocator a = new CpuAllocator();
        Tensor t = Tensor4j.builder().allocator(a).shape(4, 4).fill(2f);
        assertInstanceOf(NativeStorage.class, t.getStorage());
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                assertEquals(2f, t.get(i, j), EPS);
    }

    @Test
    void addMatchesHeapPath() {
        CpuAllocator a = new CpuAllocator();
        Tensor heapA = Tensor4j.builder().shape(4, 4).fill(3f);
        Tensor heapB = Tensor4j.builder().shape(4, 4).fill(5f);
        Tensor nativeA = Tensor4j.builder().allocator(a).shape(4, 4).fill(3f);
        Tensor nativeB = Tensor4j.builder().allocator(a).shape(4, 4).fill(5f);

        Tensor heapSum = heapA.add(heapB);
        Tensor nativeSum = nativeA.add(nativeB);

        assertInstanceOf(NativeStorage.class, nativeSum.getStorage());
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                assertEquals(heapSum.get(i, j), nativeSum.get(i, j), EPS);
    }

    @Test
    void mixedNativeHeapPromotesToNative() {
        CpuAllocator a = new CpuAllocator();
        Tensor heap = Tensor4j.builder().shape(4, 4).fill(1f);
        Tensor nat = Tensor4j.builder().allocator(a).shape(4, 4).fill(2f);
        Tensor result = nat.add(heap);
        assertInstanceOf(NativeStorage.class, result.getStorage());
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                assertEquals(3f, result.get(i, j), EPS);
    }

    @Test
    void unaryAndBinaryConstWork() {
        CpuAllocator a = new CpuAllocator();
        Tensor x = Tensor4j.builder().allocator(a).shape(2, 3)
                .fromArray(0f, 1f, 2f, 3f, 4f, 5f);
        Tensor y = x.add(10f).mul(2f);
        Tensor z = y.tanh();
        for (int i = 0; i < 6; i++) {
            float xi = i;
            float expected = (float) Math.tanh((xi + 10f) * 2f);
            int row = i / 3, col = i % 3;
            assertEquals(expected, z.get(row, col), EPS);
        }
        assertInstanceOf(NativeStorage.class, z.getStorage());
    }

    @Test
    void reductionsWork() {
        CpuAllocator a = new CpuAllocator();
        Tensor x = Tensor4j.builder().allocator(a).shape(2, 3)
                .fromArray(1f, 2f, 3f, 4f, 5f, 6f);
        assertEquals(21f, x.sum().item(), EPS);
        assertEquals(3.5f, x.mean().item(), EPS);
        Tensor sumDim0 = x.sum(0);
        assertEquals(5f, sumDim0.get(0, 0), EPS);
        assertEquals(7f, sumDim0.get(0, 1), EPS);
        assertEquals(9f, sumDim0.get(0, 2), EPS);
        assertInstanceOf(NativeStorage.class, sumDim0.getStorage());
    }

    @Test
    void matmulOnNative() {
        CpuAllocator a = new CpuAllocator();
        Tensor x = Tensor4j.builder().allocator(a).shape(2, 3)
                .fromArray(1f, 2f, 3f, 4f, 5f, 6f);
        Tensor y = Tensor4j.builder().allocator(a).shape(3, 2)
                .fromArray(7f, 8f, 9f, 10f, 11f, 12f);
        Tensor z = x.matmul(y);
        // [[58, 64], [139, 154]]
        assertEquals(58f, z.get(0, 0), EPS);
        assertEquals(64f, z.get(0, 1), EPS);
        assertEquals(139f, z.get(1, 0), EPS);
        assertEquals(154f, z.get(1, 1), EPS);
        assertInstanceOf(NativeStorage.class, z.getStorage());
    }

    @Test
    void backwardThroughNativeChain() {
        CpuAllocator a = new CpuAllocator();
        Tensor x = Tensor4j.builder().allocator(a).shape(2, 2).requiresGrad()
                .fromArray(1f, 2f, 3f, 4f);
        Tensor y = x.mul(x).sum();
        y.backward();
        // d/dx (x^2 sum) = 2x
        Tensor grad = x.getGrad();
        assertNotNull(grad);
        assertEquals(2f, grad.get(0, 0), EPS);
        assertEquals(4f, grad.get(0, 1), EPS);
        assertEquals(6f, grad.get(1, 0), EPS);
        assertEquals(8f, grad.get(1, 1), EPS);
    }

    @Test
    void inplaceOpsWorkOnNative() {
        CpuAllocator a = new CpuAllocator();
        Tensor x = Tensor4j.builder().allocator(a).shape(2, 2).fill(3f);
        x.add_(2f);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                assertEquals(5f, x.get(i, j), EPS);
    }

    @Test
    void emptyCacheReleasesAfterTensorsCollected() throws InterruptedException {
        CpuAllocator a = new CpuAllocator();
        for (int trial = 0; trial < 5; trial++) {
            Tensor t = Tensor4j.builder().allocator(a).shape(64, 64).rand();
            t.add(1f);
        }
        // give Cleaner a chance to run
        for (int i = 0; i < 50 && a.allocatedBytes() > 0; i++) {
            System.gc();
            Thread.sleep(20);
        }
        assertEquals(0, a.allocatedBytes(),
                "all blocks should have been freed once tensors became unreachable");
    }
}
