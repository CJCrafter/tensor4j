package com.cjcrafter.tensor4j;

import com.cjcrafter.tensor4j.util.SafeAutoClosable;

import java.util.SplittableRandom;

/**
 * Static utility class for tensor operations that are not instance methods.
 * Analogous to PyTorch's {@code torch} namespace.
 */
public final class Tensor4j {

    private static final ThreadLocal<Boolean> NO_GRAD = ThreadLocal.withInitial(() -> false);
    private static final ThreadLocal<SplittableRandom> RANDOM = ThreadLocal.withInitial(SplittableRandom::new);

    private Tensor4j() {
    }

    /**
     * Returns a new tensor builder. Set the shape and then generate to get
     * your tensor.
     *
     * @return The tensor builder.
     */
    public static TensorBuilder builder() {
        return new TensorBuilder();
    }

    /**
     * Generates a new tensor with the given {@code dims}. Each element in the
     * tensor will be uniformly distributed {@code [0, 1)}.
     *
     * @param dims The shape of the tensor.
     * @return The generated tensor.
     */
    public static Tensor rand(int... dims) {
        return new TensorBuilder().shape(dims).rand();
    }

    /**
     * Generates a new tensor with the given {@code dims}. Each element in the
     * tensor will be normally distributed with a mean of 0 and a standard
     * deviation of 1.
     *
     * @param dims The shape of the tensor.
     * @return The generated tensor.
     */
    public static Tensor randn(int... dims) {
        return new TensorBuilder().shape(dims).randn();
    }

    /**
     * Seeds the random number generator for the current thread.
     *
     * <p>If you are using multiple threads, you will need to call this method
     * 1 time for each thread you use.
     *
     * @param seed The number to seed the random number generator with.
     */
    public static void manualSeed(long seed) {
        RANDOM.set(new SplittableRandom(seed));
    }

    /**
     * Returns the current random instance for the current thread.
     */
    public static SplittableRandom getRandom() {
        return RANDOM.get();
    }

    /**
     * Returns {@code true} if gradients are currently enabled.
     */
    public static boolean isGradEnabled() {
        return !NO_GRAD.get();
    }

    /**
     * Disables gradient tracking in a try-with-resources block.
     *
     * <pre>{@code
     *     try (var ignored = Tensor4j.noGrad()) {
     *         // inference or optimizer steps
     *     }
     * }</pre>
     *
     * @return The closable that re-enables gradients afterward.
     */
    public static SafeAutoClosable noGrad() {
        // handle nested noGrad calls for re-entrant
        boolean prev = NO_GRAD.get();
        NO_GRAD.set(true);
        return () -> NO_GRAD.set(prev);
    }

    /**
     * Returns a random permutation of integers from 0 to n-1.
     */
    public static int[] randperm(int n) {
        SplittableRandom rng = RANDOM.get();
        int[] perm = new int[n];
        for (int i = 0; i < n; i++)
            perm[i] = i;

        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
        return perm;
    }
}
