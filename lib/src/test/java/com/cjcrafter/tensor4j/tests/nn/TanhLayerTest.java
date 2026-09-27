package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.nn.layers.Tanh;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TanhLayerTest {

    private static final float EPS = 1e-5f;

    @Test
    void forwardMatchesMathTanh() {
        Tanh tanh = new Tanh();
        Tensor x = TensorBuilder.builder().shape(1, 3).fromArray(-1f, 0f, 1f);
        Tensor y = tanh.forward(x);
        for (int i = 0; i < 3; i++)
            assertEquals((float) Math.tanh(x.get(0, i)), y.get(0, i), EPS);
    }
}
