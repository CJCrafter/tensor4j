package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.nn.layers.LayerNorm;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LayerNormTest {

    private static final float EPS = 1e-3f;

    @Test
    void normalizesToZeroMeanUnitVarianceBeforeAffine() {
        LayerNorm layerNorm = new LayerNorm(4);
        Tensor x = TensorBuilder.builder().shape(1, 4).fromArray(1f, 2f, 3f, 4f);
        Tensor y = layerNorm.forward(x);

        // default weight=1, bias=0, so output should already be normalized
        float mean = 0f;
        for (int i = 0; i < 4; i++) mean += y.get(0, i);
        mean /= 4f;
        assertEquals(0f, mean, EPS);

        float variance = 0f;
        for (int i = 0; i < 4; i++) variance += (y.get(0, i) - mean) * (y.get(0, i) - mean);
        variance /= 4f;
        assertEquals(1f, variance, EPS);
    }

    @Test
    void rejectsMismatchedLastDim() {
        LayerNorm layerNorm = new LayerNorm(4);
        Tensor x = TensorBuilder.builder().shape(1, 3).fromArray(1f, 2f, 3f);
        assertThrows(IllegalArgumentException.class, () -> layerNorm.forward(x));
    }

    @Test
    void constantInputProducesZeroBeforeBias() {
        LayerNorm layerNorm = new LayerNorm(3);
        Tensor x = TensorBuilder.builder().shape(1, 3).fill(5f);
        Tensor y = layerNorm.forward(x);
        for (int i = 0; i < 3; i++)
            assertEquals(0f, y.get(0, i), 1e-2f);
    }
}
