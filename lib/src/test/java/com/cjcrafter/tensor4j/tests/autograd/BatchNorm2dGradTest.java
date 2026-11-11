package com.cjcrafter.tensor4j.tests.autograd;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.nn.layers.BatchNorm2d;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BatchNorm2dGradTest {

    private static final float EPS = 2e-2f;
    private static final float DELTA = 1e-3f;

    @Test
    void batchNorm2dBackwardMatchesFiniteDiff() {
        Tensor4j.manualSeed(31);
        Tensor input = Tensor4j.builder().shape(3, 2, 3, 3).requiresGrad().randn();
        BatchNorm2d bn = new BatchNorm2d(2);

        Tensor loss = bn.forward(input).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
        }

        float[] grad = input.getGrad().toFloatArray();
        int n = input.getShape().numel();
        assertEquals(n, grad.length);

        for (int i = 0; i < n; i++) {
            float orig = read(input, i);
            write(input, i, orig - DELTA);
            float lo = recompute(bn, input);
            write(input, i, orig + DELTA);
            float hi = recompute(bn, input);
            write(input, i, orig);

            float numeric = (hi - lo) / (2f * DELTA);
            float analytic = grad[i];
            float tol = Math.max(EPS, EPS * Math.max(Math.abs(numeric), Math.abs(analytic)));
            assertEquals(numeric, analytic, tol,
                "grad mismatch at index " + i + ": analytic=" + analytic + " numeric=" + numeric);
        }
    }

    private static float recompute(BatchNorm2d bn, Tensor input) {
        try (var ignored = Tensor4j.noGrad()) {
            return bn.forward(input).sum().item();
        }
    }

    private static float read(Tensor t, int flatIdx) {
        return t.getStorage().array()[t.getOffset() + flatIdx];
    }

    private static void write(Tensor t, int flatIdx, float val) {
        t.getStorage().array()[t.getOffset() + flatIdx] = val;
    }
}
