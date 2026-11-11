package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.nn.losses.HuberLoss;
import com.cjcrafter.tensor4j.nn.losses.Loss;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class HuberLossTest {

    private static final float EPS = 1e-5f;

    @Test
    void noneReductionSwitchesBetweenQuadraticAndLinear() {
        // delta = 1: diff=1 is right at the boundary (quadratic), diff=5 is linear
        Tensor input = TensorBuilder.builder().shape(1, 2).fromArray(1f, 5f);
        Tensor target = TensorBuilder.builder().shape(1, 2).fromArray(0f, 0f);

        Tensor loss = new HuberLoss(Loss.Reduction.NONE).forward(input, target);
        assertEquals(0.5f, loss.get(0, 0), EPS);   // 0.5 * 1^2
        assertEquals(4.5f, loss.get(0, 1), EPS);   // 1 * (5 - 0.5)
    }

    @Test
    void meanAndSumReductions() {
        Tensor input = TensorBuilder.builder().shape(1, 2).fromArray(1f, 5f);
        Tensor target = TensorBuilder.builder().shape(1, 2).fromArray(0f, 0f);

        assertEquals(2.5f, new HuberLoss(Loss.Reduction.MEAN).forward(input, target).item(), EPS);
        assertEquals(5.0f, new HuberLoss(Loss.Reduction.SUM).forward(input, target).item(), EPS);
    }

    @Test
    void zeroErrorGivesZeroLoss() {
        Tensor input = TensorBuilder.builder().shape(1, 3).fromArray(1f, 2f, 3f);
        Tensor target = TensorBuilder.builder().shape(1, 3).fromArray(1f, 2f, 3f);
        assertEquals(0f, new HuberLoss().forward(input, target).item(), EPS);
    }
}
