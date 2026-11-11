package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.nn.layers.BatchNorm2d;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class BatchNorm2dTest {

    private static final float EPS = 1e-3f;

    @Test
    void trainModeNormalizesEachChannelToZeroMeanUnitVariance() {
        Tensor4j.manualSeed(1);
        BatchNorm2d bn = new BatchNorm2d(3);
        Tensor input = Tensor4j.builder().shape(4, 3, 5, 5).randn();
        Tensor output = bn.forward(input);

        int n = 4, h = 5, w = 5;
        for (int c = 0; c < 3; c++) {
            float sum = 0f;
            for (int ni = 0; ni < n; ni++)
                for (int hi = 0; hi < h; hi++)
                    for (int wi = 0; wi < w; wi++)
                        sum += output.get(ni, c, hi, wi);
            float mean = sum / (n * h * w);
            assertEquals(0f, mean, 1e-2f, "channel " + c + " mean should be ~0");

            float varSum = 0f;
            for (int ni = 0; ni < n; ni++)
                for (int hi = 0; hi < h; hi++)
                    for (int wi = 0; wi < w; wi++) {
                        float d = output.get(ni, c, hi, wi) - mean;
                        varSum += d * d;
                    }
            float variance = varSum / (n * h * w);
            assertEquals(1f, variance, 1e-2f, "channel " + c + " variance should be ~1");
        }
    }

    @Test
    void trainingUpdatesRunningStats() {
        Tensor4j.manualSeed(2);
        BatchNorm2d bn = new BatchNorm2d(2);
        Tensor before = bn.stateDict().get("runningMean").clone();

        Tensor input = Tensor4j.builder().shape(4, 2, 4, 4).randn();
        bn.forward(input);

        Tensor after = bn.stateDict().get("runningMean");
        boolean changed = false;
        for (int c = 0; c < 2; c++) {
            if (Math.abs(before.get(0, c, 0, 0) - after.get(0, c, 0, 0)) > 1e-6f)
                changed = true;
        }
        assertTrue(changed, "running mean should move away from its zero initialization after training");
    }

    @Test
    void evalModeUsesRunningStatsNotBatchStats() {
        Tensor4j.manualSeed(3);
        BatchNorm2d bn = new BatchNorm2d(1);

        // Run several training batches so running stats diverge from a fresh batch's stats.
        for (int i = 0; i < 5; i++) {
            bn.forward(Tensor4j.builder().shape(4, 1, 4, 4).randn());
        }

        bn.eval();
        Tensor batch = Tensor4j.builder().shape(4, 1, 4, 4).fill(100f); // wildly different from training data
        Tensor output = bn.forward(batch);

        // If eval mode were (incorrectly) using batch stats, a constant-100 input would
        // normalize to all zeros. Since it should use the running stats instead, it won't.
        boolean allZero = true;
        for (int ni = 0; ni < 4; ni++)
            for (int hi = 0; hi < 4; hi++)
                for (int wi = 0; wi < 4; wi++)
                    if (Math.abs(output.get(ni, 0, hi, wi)) > EPS) allZero = false;
        assertFalse(allZero, "eval mode should normalize using running stats, not the batch's own stats");
    }

    @Test
    void weightAndBiasAndRunningStatsAreAllRegistered() {
        Tensor4j.manualSeed(4);
        BatchNorm2d bn = new BatchNorm2d(3);
        Map<String, Tensor> state = bn.stateDict();
        assertEquals(4, state.size());
        assertTrue(state.containsKey("weight"));
        assertTrue(state.containsKey("bias"));
        assertTrue(state.containsKey("runningMean"));
        assertTrue(state.containsKey("runningVar"));

        assertTrue(state.get("weight").requiresGrad());
        assertTrue(state.get("bias").requiresGrad());
        assertFalse(state.get("runningMean").requiresGrad());
        assertFalse(state.get("runningVar").requiresGrad());
    }
}
