package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.nn.losses.CrossEntropyLoss;
import com.cjcrafter.tensor4j.nn.losses.Loss;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CrossEntropyLossTest {

    private static final float EPS = 1e-4f;

    @Test
    void matchesHandComputedNegLogLikelihoodForOneHotTarget() {
        // logits = [1, 2, 3], one-hot target on index 2
        Tensor logits = TensorBuilder.builder().shape(1, 3).fromArray(1f, 2f, 3f);
        Tensor target = TensorBuilder.builder().shape(1, 3).fromArray(0f, 0f, 1f);

        double sumExp = Math.exp(1) + Math.exp(2) + Math.exp(3);
        double expected = -(3 - Math.log(sumExp));

        Tensor loss = new CrossEntropyLoss().forward(logits, target);
        assertEquals((float) expected, loss.item(), EPS);
    }

    @Test
    void defaultDimIsLastDim() {
        Tensor logits = TensorBuilder.builder().shape(1, 3).fromArray(1f, 2f, 3f);
        Tensor target = TensorBuilder.builder().shape(1, 3).fromArray(0f, 0f, 1f);

        Tensor viaDefault = new CrossEntropyLoss().forward(logits, target);
        Tensor viaExplicit = new CrossEntropyLoss(Loss.Reduction.MEAN, 1).forward(logits, target);
        assertEquals(viaDefault.item(), viaExplicit.item(), EPS);
    }

    @Test
    void sumReductionOverBatch() {
        // two rows, each a confident one-hot prediction -> near-zero loss each
        Tensor logits = TensorBuilder.builder().shape(2, 2).fromArray(10f, -10f, -10f, 10f);
        Tensor target = TensorBuilder.builder().shape(2, 2).fromArray(1f, 0f, 0f, 1f);

        Tensor loss = new CrossEntropyLoss(Loss.Reduction.SUM).forward(logits, target);
        assertEquals(0f, loss.item(), 1e-3f);
    }
}
