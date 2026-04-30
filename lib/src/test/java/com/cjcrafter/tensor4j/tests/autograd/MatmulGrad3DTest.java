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

    private static float recompute(Tensor a, Tensor b) {
        try (var ignored = Tensor4j.noGrad()) {
            return a.matmul(b).sum().item();
        }
    }

    private static void checkGradByFiniteDiff(Tensor t, java.util.function.Supplier<Float> forward) {
        float[] data = t.getData();
        float[] grad = t.getGrad().getData();
        assertEquals(data.length, grad.length);

        for (int i = 0; i < data.length; i++) {
            float orig = data[i];
            data[i] = orig - DELTA;
            float lo = forward.get();
            data[i] = orig + DELTA;
            float hi = forward.get();
            data[i] = orig;

            float numeric = (hi - lo) / (2f * DELTA);
            float analytic = grad[i];
            float tol = Math.max(EPS, EPS * Math.max(Math.abs(numeric), Math.abs(analytic)));
            assertEquals(numeric, analytic, tol,
                "grad mismatch at index " + i + ": analytic=" + analytic + " numeric=" + numeric);
        }
    }
}
