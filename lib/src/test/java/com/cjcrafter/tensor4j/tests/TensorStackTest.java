package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorStackTest {

    private static final float EPS = 1e-6f;

    @Test
    void stackAtDimZeroAddsBatchAxis() {
        Tensor a = TensorBuilder.builder().shape(2, 3).fromArray(1f, 2f, 3f, 4f, 5f, 6f);
        Tensor b = TensorBuilder.builder().shape(2, 3).fromArray(7f, 8f, 9f, 10f, 11f, 12f);

        Tensor stacked = Tensor.stack(0, a, b);
        assertArrayEquals(new int[]{2, 2, 3}, stacked.getShape().dims());

        assertEquals(1f, stacked.get(0, 0, 0), EPS);
        assertEquals(6f, stacked.get(0, 1, 2), EPS);
        assertEquals(7f, stacked.get(1, 0, 0), EPS);
        assertEquals(12f, stacked.get(1, 1, 2), EPS);
    }

    @Test
    void stackAtLastDimAppendsAxis() {
        Tensor a = TensorBuilder.builder().shape(1, 3).fromArray(1f, 2f, 3f);
        Tensor b = TensorBuilder.builder().shape(1, 3).fromArray(4f, 5f, 6f);

        Tensor stacked = Tensor.stack(2, a, b);
        assertArrayEquals(new int[]{1, 3, 2}, stacked.getShape().dims());
        for (int i = 0; i < 3; i++) {
            assertEquals(i + 1f, stacked.get(0, i, 0), EPS);
            assertEquals(i + 4f, stacked.get(0, i, 1), EPS);
        }
    }
}
