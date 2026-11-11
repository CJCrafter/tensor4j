package com.cjcrafter.tensor4j.tests.autograd;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.autograd.MaxPool2dFn;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MaxPool2dGradTest {

    private static final float EPS = 2e-2f;
    private static final float DELTA = 1e-3f;

    @Test
    void maxPool2dBackwardMatchesFiniteDiff() {
        Tensor4j.manualSeed(21);
        // N=2, C=3, H=6, W=6; 2x2 windows, stride 2 -> 3x3 output
        Tensor input = TensorBuilder.builder().shape(2, 3, 6, 6).requiresGrad().randn();
        int kernelSize = 2, stride = 2;

        Tensor loss = MaxPool2dFn.apply(input, kernelSize, stride).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
        }

        float[] grad = input.getGrad().toFloatArray();
        int n = input.getShape().numel();
        assertEquals(n, grad.length);

        for (int i = 0; i < n; i++) {
            float orig = read(input, i);
            write(input, i, orig - DELTA);
            float lo = recompute(input, kernelSize, stride);
            write(input, i, orig + DELTA);
            float hi = recompute(input, kernelSize, stride);
            write(input, i, orig);

            float numeric = (hi - lo) / (2f * DELTA);
            float analytic = grad[i];
            float tol = Math.max(EPS, EPS * Math.max(Math.abs(numeric), Math.abs(analytic)));
            assertEquals(numeric, analytic, tol,
                "grad mismatch at index " + i + ": analytic=" + analytic + " numeric=" + numeric);
        }
    }

    private static float recompute(Tensor input, int kernelSize, int stride) {
        try (var ignored = Tensor4j.noGrad()) {
            return MaxPool2dFn.apply(input, kernelSize, stride).sum().item();
        }
    }

    private static float read(Tensor t, int flatIdx) {
        return t.getStorage().array()[t.getOffset() + flatIdx];
    }

    private static void write(Tensor t, int flatIdx, float val) {
        t.getStorage().array()[t.getOffset() + flatIdx] = val;
    }
}
