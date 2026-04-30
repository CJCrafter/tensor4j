package com.cjcrafter.tensor4j.tests.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.nn.layers.*;
import com.cjcrafter.tensor4j.util.Pair;
import com.cjcrafter.tensor4j.util.Triple;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * We should compare the analytical grad (the calculated gradient functions) to
 * the empirical grad (adding/subtracting a small {@code DELTA}, and measuring
 * the change). These 2 values should be approximately equal.
 *
 * <p>This test is <i>not great</i> for non-differentiable functions, so those
 * functions have special cases to avoid the non-differentiable areas.
 */
class GradTest {

    private static final int[] SIZES = { 1, 4, 64 };
    private static final float EPSILON = 1e-2f; // less than 1% difference
    private static final float DELTA = 1e-3f;


    private static void testElementwiseGradients(Function<Tensor[], Tensor> forward, Tensor... inputs) {
        for (Tensor input : inputs) {
            float[] param = input.getData();
            float[] grad = input.getGrad().getData();
            assertEquals(param.length, grad.length);

            for (int i = 0; i < param.length; i++) {
                float original = param[i];

                param[i] = original - DELTA;
                float lossLower = forward.apply(inputs).getData()[i];
                param[i] = original + DELTA;
                float lossUpper = forward.apply(inputs).getData()[i];
                param[i] = original;

                assertGradMatches(grad[i], lossLower, lossUpper, i, original);
            }
        }
    }

    private static void testScalarGradients(Function<Tensor[], Tensor> forward, Tensor... inputs) {
        for (Tensor input : inputs) {
            float[] param = input.getData();
            float[] grad = input.getGrad().getData();
            assertEquals(param.length, grad.length);

            for (int i = 0; i < param.length; i++) {
                float original = param[i];

                param[i] = original - DELTA;
                float lossLower = forward.apply(inputs).item();
                param[i] = original + DELTA;
                float lossUpper = forward.apply(inputs).item();
                param[i] = original;

                assertGradMatches(grad[i], lossLower, lossUpper, i, original);
            }
        }
    }

    private static void assertGradMatches(float analytic, float lossLower, float lossUpper, int i, float inputVal) {
        float numerical = (lossUpper - lossLower) / (2f * DELTA);
        float denom = Math.max(Math.abs(numerical), Math.abs(analytic));
        float relError = denom == 0f ? 0f : Math.abs(numerical - analytic) / denom;
        assertTrue(relError < EPSILON,
                "grad mismatch at index " + i
                        + ": numerical=" + numerical + " analytic=" + analytic
                        + " relError=" + relError
                        + " input value=" + inputVal);
    }

    private static Stream<Arguments> testUnaryElementwiseGrad() {
        List<Pair<Function<Tensor, Tensor>, String>> elements = Arrays.asList(
                new Pair<>(Tensor::tanh, "tanh"),
                new Pair<>(Tensor::abs, "abs"),
                new Pair<>(Tensor::exp, "exp"),
                new Pair<>(Tensor::log, "log"),
                new Pair<>(Tensor::sqrt, "sqrt"),
                new Pair<>(Tensor::square, "square")
        );

        return elements.stream().flatMap(tuple ->
                Arrays.stream(SIZES).mapToObj(size -> Arguments.of(tuple.first(), tuple.second(), size))
        );
    }

    @DisplayName("Unary Gradient Tests")
    @ParameterizedTest(name = "{1}({2}x{2})")
    @MethodSource
    void testUnaryElementwiseGrad(Function<Tensor, Tensor> forward, String opName, int size) {
        Tensor4j.manualSeed(2187);

        Tensor a = switch (opName) {
            // log/sqrt is not defined for negative values
            case "log", "sqrt" -> Tensor4j.builder().shape(size, size).requiresGrad().rand(10 * DELTA, 2f);
            // the derivative for abs is not defined for 0
            case "abs" -> {
                Tensor temp = Tensor4j.builder().shape(size, size).requiresGrad().randn();
                Tensor jitter = temp.abs().le(10 * DELTA).mul(20 * DELTA);
                yield temp.add_(jitter);
            }
            default -> Tensor4j.builder().shape(size, size).requiresGrad().randn();
        };
        Tensor c = forward.apply(a);
        Tensor sum = c.sum();

        try (var ignored = Tensor4j.noGrad()) {
            sum.backward();
            testElementwiseGradients(in -> forward.apply(in[0]), a);
        }
    }

    private static Stream<Arguments> testBinaryElementwiseGrad() {
        List<Pair<BiFunction<Tensor, Tensor, Tensor>, String>> elements = Arrays.asList(
                new Pair<>(Tensor::add, "add"),
                new Pair<>(Tensor::sub, "sub"),
                new Pair<>(Tensor::mul, "mul"),
                new Pair<>(Tensor::div, "div"),
                new Pair<>(Tensor::max, "max"),
                new Pair<>(Tensor::min, "min")
        );

        return elements.stream().flatMap(tuple ->
                Arrays.stream(SIZES).mapToObj(size -> Arguments.of(tuple.first(), tuple.second(), size))
        );
    }


    @DisplayName("Binary Gradient Tests")
    @ParameterizedTest(name = "{1}({2}x{2}, {2}x{2})")
    @MethodSource
    void testBinaryElementwiseGrad(BiFunction<Tensor, Tensor, Tensor> forward, String opName, int size) {
        Tensor4j.manualSeed(2187);

        Tensor a = Tensor4j.builder().shape(size, size).requiresGrad().randn();
        Tensor b = switch (opName) {
            // division is not defined at 0
            case "div" -> {
                Tensor temp = Tensor4j.builder().shape(size, size).requiresGrad().randn();
                Tensor jitter = temp.abs().le(10 * DELTA).mul(20 * DELTA);
                yield temp.add_(jitter);
            }
            // max/min across the boundary is not differentiable
            case "max", "min" -> {
                Tensor temp = Tensor4j.builder().shape(size, size).requiresGrad().randn();
                Tensor jitter = temp.sub(a).abs().le(10 * DELTA).mul(20 * DELTA);
                yield temp.add_(jitter);
            }
            default -> Tensor4j.builder().shape(size, size).requiresGrad().randn();
        };
        Tensor c = forward.apply(a, b);
        Tensor sum = c.sum();

        try (var ignored = Tensor4j.noGrad()) {
            sum.backward();
            testElementwiseGradients(in -> forward.apply(in[0], in[1]), a, b);
        }
    }

    private static Stream<Arguments> testBinaryConstElementwiseGrad() {
        List<Pair<BiFunction<Tensor, Float, Tensor>, String>> elements = Arrays.asList(
                new Pair<>(Tensor::add, "add"),
                new Pair<>(Tensor::sub, "sub"),
                new Pair<>(Tensor::mul, "mul"),
                new Pair<>(Tensor::div, "div"),
                new Pair<>(Tensor::max, "max"),
                new Pair<>(Tensor::min, "min"),
                new Pair<>(Tensor::rdiv, "rdiv"),
                new Pair<>(Tensor::rsub, "rsub")
        );

        return elements.stream().flatMap(tuple ->
                Arrays.stream(SIZES).mapToObj(size -> Arguments.of(tuple.first(), tuple.second(), size))
        );
    }


    @DisplayName("Binary Const Gradient Tests")
    @ParameterizedTest(name = "{1}({2}x{2}, {2}x{2})")
    @MethodSource
    void testBinaryConstElementwiseGrad(BiFunction<Tensor, Float, Tensor> forward, String opName, int size) {
        Tensor4j.manualSeed(2187);

        // rdiv computes c/a, so a must avoid 0
        Tensor a = switch (opName) {
            case "rdiv" -> {
                Tensor temp = Tensor4j.builder().shape(size, size).requiresGrad().randn();
                Tensor jitter = temp.abs().le(10 * DELTA).mul(20 * DELTA);
                yield temp.add_(jitter);
            }
            default -> Tensor4j.builder().shape(size, size).requiresGrad().randn();
        };
        // divide by 0 is not defined
        float b = switch (opName) {
            case "div" -> {
                float temp = Tensor4j.getRandom().nextFloat(-2f, 2f);
                if (Math.abs(temp) < 0.5f) {
                    temp += 0.5f;
                }
                yield temp;
            }
            default -> Tensor4j.getRandom().nextFloat(-2f, 2f);
        };

        Tensor c = forward.apply(a, b);
        Tensor sum = c.sum();

        try (var ignored = Tensor4j.noGrad()) {
            sum.backward();
            testElementwiseGradients(in -> forward.apply(in[0], b), a);
        }
    }

    private static Stream<Arguments> testReductionGrad() {
        List<Triple<BiFunction<Tensor, Integer, Tensor>, Integer, String>> elements = Arrays.asList(
                new Triple<>(Tensor::sum, 0, "sum"),
                new Triple<>(Tensor::sum, 1, "sum"),
                new Triple<>(Tensor::max, 0, "max"),
                new Triple<>(Tensor::max, 1, "max"),
                new Triple<>((t, dim) -> t.indexSelect(dim, 0, 1, 0), 0, "indexSelect"),
                new Triple<>((t, dim) -> t.indexSelect(dim, 0, 1, 0), 1, "indexSelect")
        );

        int[] reductionSizes = { 2, 4, 8 };
        return elements.stream().flatMap(tuple ->
                Arrays.stream(reductionSizes).mapToObj(size -> Arguments.of(tuple.first(), tuple.second(), tuple.third(), size))
        );
    }

    @DisplayName("Reduction Gradient Tests")
    @ParameterizedTest(name = "{2}({3}x{3}, dim={1})")
    @MethodSource
    void testReductionGrad(BiFunction<Tensor, Integer, Tensor> forward, int dim, String opName, int size) {
        Tensor4j.manualSeed(2187);

        Tensor a = Tensor4j.builder().shape(size, size).requiresGrad().randn();
        Tensor c = forward.apply(a, dim);
        Tensor sum = c.sum();

        try (var ignored = Tensor4j.noGrad()) {
            sum.backward();
            testScalarGradients(in -> forward.apply(in[0], dim).sum(), a);
        }
    }

    private static Stream<Arguments> testMatmulGrad() {
        // {M, K, N}: result of (M,K) @ (K,N) is (M,N).
        return Stream.of(
                Arguments.of(2, 3, 4),
                Arguments.of(4, 4, 4),
                Arguments.of(4, 1, 4),
                Arguments.of(1, 5, 3),
                Arguments.of(8, 3, 2)
        );
    }

    @DisplayName("Matmul Gradient Tests")
    @ParameterizedTest(name = "({0}x{1}) @ ({1}x{2})")
    @MethodSource
    void testMatmulGrad(int m, int k, int n) {
        Tensor4j.manualSeed(2187);
        Tensor a = Tensor4j.builder().shape(m, k).requiresGrad().randn();
        Tensor b = Tensor4j.builder().shape(k, n).requiresGrad().randn();

        Tensor loss = a.matmul(b).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
            testScalarGradients(in -> in[0].matmul(in[1]).sum(), a, b);
        }
    }

    @Test
    void testMeanGrad() {
        Tensor4j.manualSeed(2187);
        Tensor a = Tensor4j.builder().shape(4, 4).requiresGrad().randn();

        Tensor loss = a.mean();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
        }

        // dL/da[i] = 1/N for every element
        float expected = 1f / a.getShape().numel();
        for (float g : a.getGrad().getData()) {
            assertEquals(expected, g, 1e-6f);
        }
    }

    @Test
    void testTransposeGrad() {
        Tensor4j.manualSeed(2187);
        Tensor a = Tensor4j.builder().shape(3, 5).requiresGrad().randn();

        // a.T @ a is (5,3)@(3,5) -> (5,5); exercises transpose backward
        // *and* matmul backward through a strided view.
        Tensor loss = a.transpose().matmul(a).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
            testScalarGradients(in -> in[0].transpose().matmul(in[0]).sum(), a);
        }
    }

    @Test
    void testSelfMul() {
        Tensor4j.manualSeed(2187);
        Tensor a = Tensor4j.builder().shape(4, 4).requiresGrad().randn();

        Tensor loss = a.mul(a).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
        }

        float[] data = a.getData();
        float[] grad = a.getGrad().getData();
        for (int i = 0; i < data.length; i++) {
            assertEquals(2f * data[i], grad[i], 1e-5f,
                    "expected dL/da[" + i + "] = 2a[i] = " + (2f * data[i]));
        }
    }

    @Test
    void testFanOut() {
        Tensor4j.manualSeed(2187);
        Tensor a = Tensor4j.builder().shape(4, 4).requiresGrad().randn();
        final int K = 8;

        Tensor loss = a.mul(1f).sum();
        for (int k = 1; k < K; k++) {
            loss = loss.add(a.mul((float) (k + 1)).sum());
        }
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
        }

        float expected = K * (K + 1) / 2f;
        for (float g : a.getGrad().getData()) {
            assertEquals(expected, g, 1e-4f);
        }
    }

    @Test
    void testUnevenBranchDepths() {
        Tensor4j.manualSeed(2187);
        Tensor a = Tensor4j.builder().shape(4, 4).requiresGrad().randn();

        Tensor deep = a;
        for (int i = 0; i < 6; i++) deep = deep.tanh();
        Tensor shallow = a.mul(2f);

        Tensor loss = deep.add(shallow).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
            testScalarGradients(in -> {
                Tensor d = in[0];
                for (int i = 0; i < 6; i++) d = d.tanh();
                return d.add(in[0].mul(2f)).sum();
            }, a);
        }
    }

    @Test
    void testCatGrad() {
        Tensor4j.manualSeed(2187);
        Tensor a = Tensor4j.builder().shape(3, 4).requiresGrad().randn();
        Tensor b = Tensor4j.builder().shape(3, 2).requiresGrad().randn();
        Tensor c = Tensor4j.builder().shape(3, 5).requiresGrad().randn();
        Tensor w = Tensor4j.builder().shape(11, 1).randn();

        Tensor loss = Tensor.cat(1, a, b, c).matmul(w).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
            testScalarGradients(in -> Tensor.cat(1, in[0], in[1], in[2]).matmul(w).sum(), a, b, c);
        }
    }

    @Test
    void testCatGradDim0() {
        Tensor4j.manualSeed(2187);
        Tensor a = Tensor4j.builder().shape(2, 3).requiresGrad().randn();
        Tensor b = Tensor4j.builder().shape(4, 3).requiresGrad().randn();
        Tensor w = Tensor4j.builder().shape(3, 1).randn();

        Tensor loss = Tensor.cat(0, a, b).matmul(w).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
            testScalarGradients(in -> Tensor.cat(0, in[0], in[1]).matmul(w).sum(), a, b);
        }
    }

    @Test
    void testSoftmaxGrad() {
        Tensor4j.manualSeed(2187);
        Tensor a = Tensor4j.builder().shape(4, 4).requiresGrad().randn();

        Tensor loss = a.softmax(1).square().sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
            testScalarGradients(in -> in[0].softmax(1).square().sum(), a);
        }
    }

    @Test
    void testRowCenterAndSquare() {
        Tensor4j.manualSeed(2187);
        Tensor a = Tensor4j.builder().shape(4, 8).requiresGrad().randn();
        final float n = a.getShape().dim(1);

        Tensor centered = a.sub(a.sum(1).mul(1f / n));
        Tensor loss = centered.mul(centered).sum();
        try (var ignored = Tensor4j.noGrad()) {
            loss.backward();
            testScalarGradients(in -> {
                Tensor c = in[0].sub(in[0].sum(1).mul(1f / n));
                return c.mul(c).sum();
            }, a);
        }
    }
}
