package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SampleCategoricalTest {

    @Test
    void alwaysPicksTheOnlyNonZeroClass() {
        Tensor4j.manualSeed(1);
        Tensor probs = TensorBuilder.builder().shape(1, 4).fromArray(0f, 0f, 1f, 0f);
        for (int trial = 0; trial < 20; trial++) {
            Tensor idx = Tensor4j.sampleCategorical(probs, 1);
            assertEquals(2f, idx.get(0, 0));
        }
    }

    @Test
    void resultShapeCollapsesSampledDim() {
        Tensor4j.manualSeed(2);
        Tensor probs = TensorBuilder.builder().shape(3, 5).rand(0.1f, 1f);
        Tensor idx = Tensor4j.sampleCategorical(probs, 1);
        assertArrayEquals(new int[]{3, 1}, idx.getShape().dims());
    }

    @Test
    void skewedDistributionFavorsHeavyClass() {
        Tensor4j.manualSeed(3);
        Tensor probs = TensorBuilder.builder().shape(1, 2).fromArray(0.95f, 0.05f);

        int heavy = 0;
        int trials = 2000;
        for (int i = 0; i < trials; i++) {
            Tensor idx = Tensor4j.sampleCategorical(probs, 1);
            if (idx.get(0, 0) == 0f) heavy++;
        }
        double frequency = heavy / (double) trials;
        assertTrue(frequency > 0.85, "expected heavy class to dominate, got frequency " + frequency);
    }
}
