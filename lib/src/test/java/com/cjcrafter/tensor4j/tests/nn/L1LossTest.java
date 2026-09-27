package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.nn.losses.L1Loss;
import com.cjcrafter.tensor4j.nn.losses.Loss;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class L1LossTest {

    private static final float EPS = 1e-5f;

    @Test
    void meanReductionIsDefault() {
        Tensor input = TensorBuilder.builder().shape(1, 3).fromArray(1f, 2f, 3f);
        Tensor target = TensorBuilder.builder().shape(1, 3).fromArray(2f, 2f, 6f);
        // |1-2| + |2-2| + |3-6| = 1 + 0 + 3 = 4, mean = 4/3
        assertEquals(4f / 3f, new L1Loss().forward(input, target).item(), EPS);
    }

    @Test
    void sumReduction() {
        Tensor input = TensorBuilder.builder().shape(1, 3).fromArray(1f, 2f, 3f);
        Tensor target = TensorBuilder.builder().shape(1, 3).fromArray(2f, 2f, 6f);
        assertEquals(4f, new L1Loss(Loss.Reduction.SUM).forward(input, target).item(), EPS);
    }

    @Test
    void noneReductionKeepsShape() {
        Tensor input = TensorBuilder.builder().shape(1, 3).fromArray(1f, 2f, 3f);
        Tensor target = TensorBuilder.builder().shape(1, 3).fromArray(2f, 2f, 6f);
        Tensor result = new L1Loss(Loss.Reduction.NONE).forward(input, target);
        assertEquals(1f, result.get(0, 0), EPS);
        assertEquals(0f, result.get(0, 1), EPS);
        assertEquals(3f, result.get(0, 2), EPS);
    }
}
