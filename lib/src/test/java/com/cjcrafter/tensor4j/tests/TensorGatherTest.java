package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorGatherTest {

    private static final float EPS = 1e-6f;

    @Test
    void selectsOneElementPerRow() {
        Tensor x = TensorBuilder.builder().shape(2, 3).fromArray(1f, 2f, 3f, 4f, 5f, 6f);
        Tensor indices = TensorBuilder.builder().shape(2, 1).fromArray(2f, 0f);

        Tensor result = x.gather(1, indices);
        assertEquals(2, result.getShape().dim(0));
        assertEquals(1, result.getShape().dim(1));
        assertEquals(3f, result.get(0, 0), EPS);
        assertEquals(4f, result.get(1, 0), EPS);
    }

    @Test
    void backwardRoutesGradientOnlyToGatheredPosition() {
        Tensor x = TensorBuilder.builder().shape(2, 3).requiresGrad()
                .fromArray(1f, 2f, 3f, 4f, 5f, 6f);
        Tensor indices = TensorBuilder.builder().shape(2, 1).fromArray(2f, 0f);

        Tensor loss = x.gather(1, indices).sum();
        loss.backward();

        Tensor grad = x.getGrad();
        // row 0: gathered index 2
        assertEquals(0f, grad.get(0, 0), EPS);
        assertEquals(0f, grad.get(0, 1), EPS);
        assertEquals(1f, grad.get(0, 2), EPS);
        // row 1: gathered index 0
        assertEquals(1f, grad.get(1, 0), EPS);
        assertEquals(0f, grad.get(1, 1), EPS);
        assertEquals(0f, grad.get(1, 2), EPS);
    }
}
