package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.nn.layers.MaxPool2d;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class MaxPool2dTest {

    private static final float EPS = 1e-6f;

    @Test
    void defaultStrideEqualsKernelSize() {
        Tensor input = Tensor4j.builder().shape(1, 1, 4, 4).fromArray(
                1, 3, 2, 4,
                5, 6, 8, 7,
                9, 12, 11, 10,
                13, 14, 16, 15);

        Tensor out = new MaxPool2d(2).forward(input);
        assertArrayEquals(new int[]{1, 1, 2, 2}, out.getShape().dims());
        assertEquals(6f, out.get(0, 0, 0, 0), EPS);
        assertEquals(8f, out.get(0, 0, 0, 1), EPS);
        assertEquals(14f, out.get(0, 0, 1, 0), EPS);
        assertEquals(16f, out.get(0, 0, 1, 1), EPS);
    }
}
