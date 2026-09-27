package com.cjcrafter.tensor4j.tests.memory;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.memory.HeapAllocator;
import com.cjcrafter.tensor4j.memory.HeapStorage;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Exercises the full pipeline: TensorBuilder -> HeapStorage tensors,
 * dispatched through Ops -> CpuBackend -> CpuF32Kernels.
 */
class TensorE2ETest {

    private static final float EPS = 1e-5f;

    @Test
    void builderProducesUsableStorageWithCustomAllocator() {
        HeapAllocator a = new HeapAllocator();
        Tensor t = Tensor4j.builder().allocator(a).shape(4, 4).fill(2f);
        assertInstanceOf(HeapStorage.class, t.getStorage());
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                assertEquals(2f, t.get(i, j), EPS);
    }

    @Test
    void unaryAndBinaryConstWork() {
        Tensor x = Tensor4j.builder().shape(2, 3)
                .fromArray(0f, 1f, 2f, 3f, 4f, 5f);
        Tensor y = x.add(10f).mul(2f);
        Tensor z = y.tanh();
        for (int i = 0; i < 6; i++) {
            float xi = i;
            float expected = (float) Math.tanh((xi + 10f) * 2f);
            int row = i / 3, col = i % 3;
            assertEquals(expected, z.get(row, col), EPS);
        }
    }

    @Test
    void reductionsWork() {
        Tensor x = Tensor4j.builder().shape(2, 3)
                .fromArray(1f, 2f, 3f, 4f, 5f, 6f);
        assertEquals(21f, x.sum().item(), EPS);
        assertEquals(3.5f, x.mean().item(), EPS);
        Tensor sumDim0 = x.sum(0);
        assertEquals(5f, sumDim0.get(0, 0), EPS);
        assertEquals(7f, sumDim0.get(0, 1), EPS);
        assertEquals(9f, sumDim0.get(0, 2), EPS);
    }

    @Test
    void matmulWorks() {
        Tensor x = Tensor4j.builder().shape(2, 3)
                .fromArray(1f, 2f, 3f, 4f, 5f, 6f);
        Tensor y = Tensor4j.builder().shape(3, 2)
                .fromArray(7f, 8f, 9f, 10f, 11f, 12f);
        Tensor z = x.matmul(y);
        // [[58, 64], [139, 154]]
        assertEquals(58f, z.get(0, 0), EPS);
        assertEquals(64f, z.get(0, 1), EPS);
        assertEquals(139f, z.get(1, 0), EPS);
        assertEquals(154f, z.get(1, 1), EPS);
    }

    @Test
    void backwardThroughChain() {
        Tensor x = Tensor4j.builder().shape(2, 2).requiresGrad()
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
    void inplaceOpsWork() {
        Tensor x = Tensor4j.builder().shape(2, 2).fill(3f);
        x.add_(2f);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                assertEquals(5f, x.get(i, j), EPS);
    }
}
