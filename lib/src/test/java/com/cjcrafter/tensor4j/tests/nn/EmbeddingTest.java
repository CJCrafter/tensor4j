package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.nn.layers.Embedding;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingTest {

    private static final float EPS = 1e-6f;

    @Test
    void forwardLooksUpRowsByIndex() {
        Tensor4j.manualSeed(5);
        Embedding embedding = new Embedding(5, 3);
        Tensor weight = embedding.stateDict().values().iterator().next();

        Tensor indices = TensorBuilder.builder().shape(1, 2).fromArray(0f, 2f);
        Tensor result = embedding.forward(indices);

        assertArrayEquals(new int[]{1, 2, 3}, result.getShape().dims());
        for (int d = 0; d < 3; d++) {
            assertEquals(weight.get(0, d), result.get(0, 0, d), EPS);
            assertEquals(weight.get(2, d), result.get(0, 1, d), EPS);
        }
    }

    @Test
    void weightIsRegisteredAsAParameter() {
        Embedding embedding = new Embedding(4, 2);
        Map<String, Tensor> state = embedding.stateDict();
        assertEquals(1, state.size());
        Tensor weight = state.values().iterator().next();
        assertTrue(weight.requiresGrad());
        assertArrayEquals(new int[]{4, 2}, weight.getShape().dims());
    }
}
