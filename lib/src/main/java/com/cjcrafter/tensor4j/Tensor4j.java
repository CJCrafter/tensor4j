package com.cjcrafter.tensor4j;

import com.cjcrafter.tensor4j.util.SafeAutoClosable;
import com.google.common.base.Preconditions;

import java.util.SplittableRandom;

/**
 * Static utility class for tensor operations that are not instance methods.
 * Analogous to PyTorch's {@code torch} namespace.
 */
public final class Tensor4j {

    /**
     *
     */
    private static final ThreadLocal<Boolean> NO_GRAD = ThreadLocal.withInitial(() -> false);

    /**
     *
     */
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
     * Returns a new tensor of zeros with the same shape as {@code like}.
      *
     * @param like the tensor whose shape to copy
     * @return the resulting tensor
     */
    public static Tensor zerosLike(Tensor like) {
        return new TensorBuilder().like(like).zeros();
    }

    /**
     * Returns a new tensor of ones with the same shape as {@code like}.
      *
     * @param like the tensor whose shape to copy
     * @return the resulting tensor
     */
    public static Tensor onesLike(Tensor like) {
        return new TensorBuilder().like(like).ones();
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
     *
     * @return the random generator
     */
    public static SplittableRandom getRandom() {
        return RANDOM.get();
    }

    /**
     * Returns {@code true} if gradients are currently enabled.
      *
     * @return the computed value
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
     *
     * @param n the number of elements
     * @return the computed indices
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

    /**
     * Samples one index per "row" from the categorical distribution along {@code dim},
     * weighted by {@code probs} (need not sum to 1; e.g. raw visit counts work fine). Not
     * differentiable. Returns an index tensor shaped like {@code probs} with {@code dim}
     * collapsed to 1 (the same shape {@link Tensor#argmax(int)} returns).
     *
     * <p>For temperature-scaled sampling (e.g. AlphaZero-style move selection from visit
     * counts), scale {@code probs} beforehand with {@code probs.log().mul(1f / temperature).exp()}.
      *
     * @param probs the probabilities to sample from; need not sum to 1
     * @param dim the dimension to operate along
     * @return the resulting tensor
     */
    public static Tensor sampleCategorical(Tensor probs, int dim) {
        int[] dims = probs.getShape().dims();
        int ndim = dims.length;
        Preconditions.checkArgument(dim >= 0 && dim < ndim,
                "dim %s out of range for %sD tensor", dim, ndim);

        int outer = 1;
        for (int d = 0; d < dim; d++) outer *= dims[d];
        int reduceDim = dims[dim];
        int inner = 1;
        for (int d = dim + 1; d < ndim; d++) inner *= dims[d];

        float[] data = probs.toFloatArray();
        SplittableRandom rng = getRandom();

        float[] outIndices = new float[outer * inner];
        for (int o = 0; o < outer; o++) {
            int base = o * reduceDim * inner;
            for (int i = 0; i < inner; i++) {
                float total = 0f;
                for (int r = 0; r < reduceDim; r++)
                    total += data[base + r * inner + i];

                float draw = (float) (rng.nextDouble() * total);
                float cumulative = 0f;
                int chosen = reduceDim - 1; // fallback for floating-point rounding
                for (int r = 0; r < reduceDim; r++) {
                    cumulative += data[base + r * inner + i];
                    if (draw < cumulative) {
                        chosen = r;
                        break;
                    }
                }
                outIndices[o * inner + i] = chosen;
            }
        }

        int[] outDims = dims.clone();
        outDims[dim] = 1;
        return TensorBuilder.builder().shape(outDims).fromArray(outIndices);
    }
}
