package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorClampTest {

    private static final float EPS = 1e-6f;

    @Test
    void clampBoundsValuesOnBothSides() {
        Tensor x = TensorBuilder.builder().shape(1, 4).fromArray(-5f, -1f, 1f, 5f);
        Tensor y = x.clamp(-2f, 2f);
        assertEquals(-2f, y.get(0, 0), EPS);
        assertEquals(-1f, y.get(0, 1), EPS);
        assertEquals(1f, y.get(0, 2), EPS);
        assertEquals(2f, y.get(0, 3), EPS);
    }

    @Test
    void clampInPlaceMutatesTensor() {
        Tensor x = TensorBuilder.builder().shape(1, 4).fromArray(-5f, -1f, 1f, 5f);
        x.clamp_(-2f, 2f);
        assertEquals(-2f, x.get(0, 0), EPS);
        assertEquals(-1f, x.get(0, 1), EPS);
        assertEquals(1f, x.get(0, 2), EPS);
        assertEquals(2f, x.get(0, 3), EPS);
    }
}
