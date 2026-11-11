package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.nn.optimizers.Optimizer;
import com.cjcrafter.tensor4j.nn.optimizers.SGD;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class OptimizerUtilsTest {

    private static final float EPS = 1e-5f;

    @Test
    void clipGradNormScalesDownLargeGradients() {
        Tensor p = TensorBuilder.builder().shape(1, 2).requiresGrad().fromArray(0f, 0f);
        // loss = 3*p0 + 4*p1, so grad = (3, 4) with norm 5
        Tensor w = TensorBuilder.builder().shape(1, 2).fromArray(3f, 4f);
        p.mul(w).sum().backward();

        float norm = Optimizer.clipGradNorm(List.of(p), 1f);
        assertEquals(5f, norm, EPS);
        assertEquals(0.6f, p.getGrad().get(0, 0), 1e-4f);
        assertEquals(0.8f, p.getGrad().get(0, 1), 1e-4f);
    }

    @Test
    void clipGradNormLeavesSmallGradientsAlone() {
        Tensor p = TensorBuilder.builder().shape(1, 2).requiresGrad().fromArray(0f, 0f);
        Tensor w = TensorBuilder.builder().shape(1, 2).fromArray(0.3f, 0.4f);
        p.mul(w).sum().backward();

        Optimizer.clipGradNorm(List.of(p), 10f);
        assertEquals(0.3f, p.getGrad().get(0, 0), EPS);
        assertEquals(0.4f, p.getGrad().get(0, 1), EPS);
    }

    @Test
    void setLearningRateChangesSgdStep() {
        Tensor p = TensorBuilder.builder().shape(1, 1).requiresGrad().singleton(1f);
        p.sum().backward();
        Optimizer sgd = new SGD(List.of(p), 0.5f);
        sgd.setLearningRate(0.1f);
        try (var ignored = com.cjcrafter.tensor4j.Tensor4j.noGrad()) {
            sgd.step();
        }
        // grad is 1, lr 0.1 -> 1 - 0.1
        assertEquals(0.9f, p.get(0, 0), EPS);
    }
}
