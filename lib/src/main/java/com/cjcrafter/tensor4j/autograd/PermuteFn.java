package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.google.common.base.Preconditions;

import java.util.Arrays;

/**
 * Backward for {@link Tensor#permute(int...)}: the upstream gradient is permuted back by the
 * inverse permutation.
 */
public class PermuteFn extends TensorFunction {

    private final int[] perm;

    /**
     * Reorders the dimensions of {@code a} by {@code perm} (sharing storage) and records the
     * backward function.
     */
    public static Tensor apply(Tensor a, int... perm) {
        int n = a.getShape().dimensions();
        Preconditions.checkArgument(perm.length == n,
                "permute expects %s dimensions, got %s", n, Arrays.toString(perm));
        boolean[] seen = new boolean[n];
        for (int p : perm) {
            Preconditions.checkArgument(p >= 0 && p < n && !seen[p],
                    "permute argument must be a permutation of 0..%s, got %s", n - 1, Arrays.toString(perm));
            seen[p] = true;
        }

        int[] dims = a.getShape().dims();
        int[] strides = a.getShape().strides();
        int[] newDims = new int[n];
        int[] newStrides = new int[n];
        for (int i = 0; i < n; i++) {
            newDims[i] = dims[perm[i]];
            newStrides[i] = strides[perm[i]];
        }
        return Tensor.wrapResult(a.getStorage(), new Shape(newDims, newStrides), a.getOffset(),
                new PermuteFn(a, perm));
    }

    /**
     * Records the backward function for permuting {@code input} by {@code perm}.
     */
    public PermuteFn(Tensor input, int... perm) {
        super(input);
        this.perm = perm.clone();
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        int[] inverse = new int[perm.length];
        for (int i = 0; i < perm.length; i++) inverse[perm[i]] = i;
        return new Tensor[]{ gradOutput.permute(inverse).contiguous() };
    }
}
