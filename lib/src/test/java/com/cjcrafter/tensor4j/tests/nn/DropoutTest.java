package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.nn.layers.Dropout;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DropoutTest {

    private static final float EPS = 1e-5f;

    @Test
    void evalModeIsIdentity() {
        Tensor4j.manualSeed(1);
        Tensor x = TensorBuilder.builder().shape(4, 4).randn();
        Dropout dropout = new Dropout(0.5f);
        dropout.eval();
        Tensor y = dropout.forward(x);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                assertEquals(x.get(i, j), y.get(i, j), EPS);
    }

    @Test
    void pAtLeastOneZeroesEverything() {
        Tensor4j.manualSeed(2);
        Tensor x = TensorBuilder.builder().shape(4, 4).fill(3f);
        Dropout dropout = new Dropout(1f);
        Tensor y = dropout.forward(x);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                assertEquals(0f, y.get(i, j), EPS);
    }

    @Test
    void trainModeSurvivorsAreScaledAndOthersZeroed() {
        Tensor4j.manualSeed(3);
        float p = 0.4f;
        float scale = 1f / (1f - p);
        Tensor x = TensorBuilder.builder().shape(16, 16).fill(2f);
        Dropout dropout = new Dropout(p);
        Tensor y = dropout.forward(x);

        boolean sawZero = false;
        boolean sawScaled = false;
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                float v = y.get(i, j);
                if (v == 0f) {
                    sawZero = true;
                } else {
                    assertEquals(2f * scale, v, EPS);
                    sawScaled = true;
                }
            }
        }
        assertTrue(sawZero, "expected at least one dropped element across 256 samples");
        assertTrue(sawScaled, "expected at least one surviving element across 256 samples");
    }
}
