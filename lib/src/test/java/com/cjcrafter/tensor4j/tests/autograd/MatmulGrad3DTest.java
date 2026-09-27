package com.cjcrafter.tensor4j.tests.autograd;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatmulGrad3DTest {

    private static final float EPS = 1e-2f;
    private static final float DELTA = 1e-3f;

    @Test
    @DisplayName("3D @ 2D backward: grads match numerical diff")
    void threeDTimes2DBackward() {
        Tensor4j.manualSeed(7);
        Tensor a = TensorBuilder.builder().shape(2, 3, 4).requiresGrad().randn();
        Tensor b = TensorBuilder.builder().shape(4, 5).requiresGrad().randn();

        Tensor loss = a.matmul(b).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
        }

        checkGradByFiniteDiff(a, () -> recompute(a, b));
        checkGradByFiniteDiff(b, () -> recompute(a, b));
    }

    @Test
    @DisplayName("2D @ 3D backward: grads match numerical diff")
    void twoDTimes3DBackward() {
        Tensor4j.manualSeed(8);
        Tensor a = TensorBuilder.builder().shape(3, 4).requiresGrad().randn();
        Tensor b = TensorBuilder.builder().shape(2, 4, 5).requiresGrad().randn();

        Tensor loss = a.matmul(b).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
        }

        checkGradByFiniteDiff(a, () -> recompute(a, b));
        checkGradByFiniteDiff(b, () -> recompute(a, b));
    }

    @Test
    @DisplayName("4D @ 3D batched broadcast backward: grads match numerical diff")
    void fourDTimesThreeDBroadcastBackward() {
        Tensor4j.manualSeed(9);
        // (j=2, 1, n=3, m=4) @ (k=2, m=4, p=5) -> (2, 2, 3, 5)
        Tensor a = TensorBuilder.builder().shape(2, 1, 3, 4).requiresGrad().randn();
        Tensor b = TensorBuilder.builder().shape(2, 4, 5).requiresGrad().randn();

        Tensor loss = a.matmul(b).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
        }

        checkGradByFiniteDiff(a, () -> recompute(a, b));
        checkGradByFiniteDiff(b, () -> recompute(a, b));
    }

    private static float recompute(Tensor a, Tensor b) {
        try (var ignored = Tensor4j.noGrad()) {
            return a.matmul(b).sum().item();
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
