package com.cjcrafter.tensor4j;

import com.cjcrafter.tensor4j.memory.Allocator;
import com.cjcrafter.tensor4j.memory.CpuAllocator;
import com.cjcrafter.tensor4j.memory.NativeStorage;
import com.google.common.base.Preconditions;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.random.RandomGenerator;

/**
 * A builder pattern to construct a {@link Tensor}.
 */
public final class TensorBuilder {

    private Shape shape;
    private boolean requiresGrad;
    private Allocator allocator = CpuAllocator.instance();

    TensorBuilder() {
    }

    /** Overrides the default {@link CpuAllocator} for this build. */
    public TensorBuilder allocator(Allocator allocator) {
        this.allocator = Preconditions.checkNotNull(allocator, "allocator");
        return this;
    }

    private Tensor build(FloatGenerator gen) {
        Preconditions.checkNotNull(shape, "shape");

        int n = shape.numel();
        NativeStorage ns = NativeStorage.allocate(allocator, DType.F32, n);
        MemorySegment seg = ns.segment();
        for (int i = 0; i < n; i++)
            seg.setAtIndex(ValueLayout.JAVA_FLOAT, i, gen.at(i));
        return new Tensor(ns, shape, 0, requiresGrad);
    }

    @FunctionalInterface
    private interface FloatGenerator {
        float at(int idx);
    }

    /**
     * Shapes this tensor like the given tensor.
     *
     * @param tensor The tensor to copy the shape from.
     * @return A non-null reference to this (builder pattern).
     * @throws IllegalStateException if the shape was already set.
     */
    public TensorBuilder like(Tensor tensor) {
        Preconditions.checkState(shape == null, "shape already set");

        this.shape = tensor.getShape();
        return this;
    }

    /**
     * Sets the dimension of this tensor.
     *
     * @param dims The dimensions of this tensor.
     * @return A non-null reference to this (builder pattern).
     * @throws IllegalArgumentException If {@code dims.length < 2}
     * @throws IllegalStateException If the shape was already set.
     */
    public TensorBuilder shape(int... dims) {
        Preconditions.checkState(shape == null, "shape already set");
        Preconditions.checkNotNull(dims, "dims");
        Preconditions.checkArgument(dims.length >= 2, "Must use at least 2 dimensions");

        this.shape = new Shape(dims);
        return this;
    }

    /**
     * Sets the shape of this tensor.
     *
     * @param shape The shape to use in this tensor.
     * @return A non-null reference to this (builder pattern).
     * @throws IllegalStateException If the shape was already set.
     */
    public TensorBuilder shape(Shape shape) {
        Preconditions.checkState(this.shape == null, "shape already set");

        this.shape = shape;
        return this;
    }

    /**
     * Sets the generated tensor to build the backpropagation graph (autograd).
     *
     * @return A non-null reference to this (builder pattern).
     */
    public TensorBuilder requiresGrad() {
        this.requiresGrad = true;
        return this;
    }

    /**
     * Sets the shape to be a {@code 1 x 1} tensor, and fills it with the given
     * value.
     *
     * @param value The value to set into the tensor.
     * @return The generated tensor.
     * @throws IllegalStateException if a shape was already set, and it was not a 1x1.
     */
    public Tensor singleton(float value) {
        Shape expectedShape = new Shape(1, 1);
        Preconditions.checkState(shape == null || shape.equals(expectedShape), "Cannot create a singleton tensor from shape %s", shape);

        if (shape == null)
            shape = expectedShape;
        return build(i -> value);
    }

    /**
     * Fills the tensor with the given array.
     *
     * @param data The array.
     * @return The generated tensor.
     * @throws IllegalArgumentException If the array length does not match the tensor numel.
     * @throws IllegalStateException If the shape was not set yet.
     */
    public Tensor fromArray(float... data) {
        if (shape == null)
            throw new IllegalStateException("shape was null, set that first");
        if (data.length != shape.numel())
            throw new IllegalArgumentException("data did not match shape " + shape);

        NativeStorage ns = NativeStorage.allocate(allocator, DType.F32, data.length);
        MemorySegment.copy(data, 0, ns.segment(), ValueLayout.JAVA_FLOAT, 0L, data.length);
        return new Tensor(ns, shape, 0, requiresGrad);
    }

    /**
     * Creates a tensor filled with zeros.
     *
     * @return The generated tensor.
     * @throws IllegalStateException If the shape was not set yet.
     */
    public Tensor zeros() {
        return build(i -> 0f);
    }

    /**
     * Creates a tensor filled with ones.
     *
     * @return The generated tensor.
     * @throws IllegalStateException If the shape was not set yet.
     */
    public Tensor ones() {
        return build(i -> 1f);
    }

    /**
     * Creates a tensor filled with a constant value.
     *
     * @param c The constant value to fill the tensor with.
     * @return The generated tensor.
     * @throws IllegalStateException If the shape was not set yet.
     */
    public Tensor fill(float c) {
        return build(i -> c);
    }

    /**
     * Creates a tensor filled with uniform random values in {@code [0, 1)}
     * using the default random generator.
     *
     * @return The generated tensor.
     * @throws IllegalStateException If the shape was not set yet.
     * @see Tensor4j#getRandom()
     */
    public Tensor rand() {
        return rand(Tensor4j.getRandom(), 0f, 1f);
    }

    /**
     * Creates a tensor filled with uniform random values in {@code [min, max)}
     * using the default random generator.
     *
     * @param min The minimum value (inclusive).
     * @param max The maximum value (exclusive).
     * @return The generated tensor.
     * @throws IllegalStateException If the shape was not set yet.
     */
    public Tensor rand(float min, float max) {
        return rand(Tensor4j.getRandom(), min, max);
    }

    /**
     * Creates a tensor filled with uniform random values in {@code [0, 1)}
     * using the given random generator.
     *
     * @param rng The random generator to use.
     * @return The generated tensor.
     * @throws IllegalStateException If the shape was not set yet.
     */
    public Tensor rand(RandomGenerator rng) {
        return rand(rng, 0f, 1f);
    }

    /**
     * Creates a tensor filled with uniform random values in {@code [min, max)}
     * using the given random generator.
     *
     * @param rng The random generator to use.
     * @param min The minimum value (inclusive).
     * @param max The maximum value (exclusive).
     * @return The generated tensor.
     * @throws IllegalStateException If the shape was not set yet.
     */
    public Tensor rand(RandomGenerator rng, float min, float max) {
        return build(i -> rng.nextFloat(min, max));
    }

    /**
     * Creates a tensor filled with values sampled from a standard normal
     * distribution (mean 0, stddev 1) using the default random generator.
     *
     * @return The generated tensor.
     * @throws IllegalStateException If the shape was not set yet.
     * @see Tensor4j#getRandom()
     */
    public Tensor randn() {
        return randn(Tensor4j.getRandom(), 0f, 1f);
    }

    /**
     * Creates a tensor filled with values sampled from a normal distribution
     * with the given mean and standard deviation. Uses the default random
     * number generator.
     *
     * @param mean The mean of the normal distribution.
     * @param stddev The standard deviation of the normal distribution.
     * @return The generated tensor.
     * @throws IllegalStateException If the shape was not set yet.
     */
    public Tensor randn(float mean, float stddev) {
        return randn(Tensor4j.getRandom(), mean, stddev);
    }

    /**
     * Creates a tensor filled with values sampled from a standard normal
     * distribution (mean 0, stddev 1) using the given random generator.
     *
     * @param rng The random generator to use.
     * @return The generated tensor.
     * @throws IllegalStateException If the shape was not set yet.
     */
    public Tensor randn(RandomGenerator rng) {
        return randn(rng, 0f, 1f);
    }

    /**
     * Creates a tensor filled with values sampled from a normal distribution
     * with the given mean and standard deviation.
     *
     * @param rng The random generator to use.
     * @param mean The mean of the normal distribution.
     * @param stddev The standard deviation of the normal distribution.
     * @return The generated tensor.
     * @throws IllegalStateException If the shape was not set yet.
     */
    public Tensor randn(RandomGenerator rng, float mean, float stddev) {
        return build(i -> mean + stddev * (float) rng.nextGaussian());
    }

    /**
     * Creates a tensor filled with values sampled from a bernoulli
     * distribution (e.g. {@code data[i] = 1f} with probability {@code p}, and
     * {@code data[i] = 0f} with probability {@code 1 - p}). Uses the default
     * random number generator.
     *
     * @param p The chance of an element being 1f, else 0f.
     * @return The generated tensor.
     * @throws IllegalStateException IF the shape was not set yet.
     */
    public Tensor bernoulli(float p) {
        return bernoulli(Tensor4j.getRandom(), p);
    }

    /**
     * Creates a tensor filled with values sampled from a bernoulli
     * distribution (e.g. {@code data[i] = 1f} with probability {@code p}, and
     * {@code data[i] = 0f} with probability {@code 1 - p}).
     *
     * @param rng The random generator to use.
     * @param p The chance of an element being 1f, else 0f.
     * @return The generated tensor.
     * @throws IllegalStateException IF the shape was not set yet.
     */
    public Tensor bernoulli(RandomGenerator rng, float p) {
        return build(i -> rng.nextFloat(0f, 1f) < p ? 1f : 0f);
    }

    /**
     * Creates a new {@link TensorBuilder} instance.
     *
     * @return A new builder.
     */
    public static TensorBuilder builder() {
        return new TensorBuilder();
    }
}
