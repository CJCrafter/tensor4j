package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.nn.layers.Softmax;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SoftmaxTest {

    private static final float EPS = 1e-5f;

    @Test
    void rowsSumToOne() {
        Softmax softmax = new Softmax(1);
        Tensor x = TensorBuilder.builder().shape(2, 3).fromArray(1f, 2f, 3f, -1f, 0f, 1f);
        Tensor y = softmax.forward(x);

        for (int row = 0; row < 2; row++) {
            float sum = 0f;
            for (int col = 0; col < 3; col++) sum += y.get(row, col);
            assertEquals(1f, sum, EPS);
        }
    }

    @Test
    void matchesHandComputedValues() {
        Softmax softmax = new Softmax(1);
        Tensor x = TensorBuilder.builder().shape(1, 3).fromArray(0f, 0f, 0f);
        Tensor y = softmax.forward(x);
        for (int i = 0; i < 3; i++)
            assertEquals(1f / 3f, y.get(0, i), EPS);
    }
}
