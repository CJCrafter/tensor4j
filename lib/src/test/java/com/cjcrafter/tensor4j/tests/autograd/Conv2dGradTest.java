package com.cjcrafter.tensor4j.tests.autograd;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.autograd.Conv2dFn;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Conv2dGradTest {

    private static final float EPS = 2e-2f;
    private static final float DELTA = 1e-3f;

    @Test
    @DisplayName("conv2d with 'same' padding: grads for input/weight/bias match numerical diff")
    void conv2dBackwardMatchesFiniteDiff() {
        Tensor4j.manualSeed(11);
        // N=2, Cin=2, H=5, W=5; Cout=3, 3x3 kernel, stride=1, padding=1 ("same")
        Tensor input = TensorBuilder.builder().shape(2, 2, 5, 5).requiresGrad().randn();
        Tensor weight = TensorBuilder.builder().shape(3, 2, 3, 3).requiresGrad().randn();
        Tensor bias = TensorBuilder.builder().shape(1, 3).requiresGrad().randn();
        int stride = 1, padding = 1;

        Tensor loss = Conv2dFn.apply(input, weight, bias, stride, padding).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
        }

        checkGradByFiniteDiff(input, () -> recompute(input, weight, bias, stride, padding));
        checkGradByFiniteDiff(weight, () -> recompute(input, weight, bias, stride, padding));
        checkGradByFiniteDiff(bias, () -> recompute(input, weight, bias, stride, padding));
    }

    @Test
    @DisplayName("conv2d with stride 2, no padding: grads for input/weight match numerical diff")
    void stridedConv2dBackwardMatchesFiniteDiff() {
        Tensor4j.manualSeed(12);
        // N=1, Cin=1, H=6, W=6; Cout=2, 3x3 kernel, stride=2, padding=0
        Tensor input = TensorBuilder.builder().shape(1, 1, 6, 6).requiresGrad().randn();
        Tensor weight = TensorBuilder.builder().shape(2, 1, 3, 3).requiresGrad().randn();
        Tensor bias = TensorBuilder.builder().shape(1, 2).requiresGrad().randn();
        int stride = 2, padding = 0;

        Tensor loss = Conv2dFn.apply(input, weight, bias, stride, padding).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
        }

        checkGradByFiniteDiff(input, () -> recompute(input, weight, bias, stride, padding));
        checkGradByFiniteDiff(weight, () -> recompute(input, weight, bias, stride, padding));
    }

    private static float recompute(Tensor input, Tensor weight, Tensor bias, int stride, int padding) {
        try (var ignored = Tensor4j.noGrad()) {
            return Conv2dFn.apply(input, weight, bias, stride, padding).sum().item();
        }
    }

    private static float read(Tensor t, int flatIdx) {
        return t.getStorage().array()[t.getOffset() + flatIdx];
    }

    private static void write(Tensor t, int flatIdx, float val) {
        t.getStorage().array()[t.getOffset() + flatIdx] = val;
    }

    private static void checkGradByFiniteDiff(Tensor t, java.util.function.Supplier<Float> forward) {
        float[] grad = t.getGrad().toFloatArray();
        int n = t.getShape().numel();
        assertEquals(n, grad.length);

        for (int i = 0; i < n; i++) {
            float orig = read(t, i);
            write(t, i, orig - DELTA);
            float lo = forward.get();
            write(t, i, orig + DELTA);
            float hi = forward.get();
            write(t, i, orig);

            float numeric = (hi - lo) / (2f * DELTA);
            float analytic = grad[i];
            float tol = Math.max(EPS, EPS * Math.max(Math.abs(numeric), Math.abs(analytic)));
            assertEquals(numeric, analytic, tol,
                "grad mismatch at index " + i + ": analytic=" + analytic + " numeric=" + numeric);
        }
    }
}
