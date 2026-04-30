package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;

import java.util.Arrays;

/**
 * Base class for all autograd operations. Each op records which tensors
 * were its inputs, and knows how to compute gradients given the upstream
 * gradient (dL/dOutput).
 *
 * <p>Subclasses override {@link #backward} to implement the chain rule
 * for their specific operation.
 */
public abstract class TensorFunction {

    protected final Tensor[] inputs;

    protected TensorFunction(Tensor... inputs) {
        this.inputs = inputs;
    }

    public Tensor[] getInputs() {
        return inputs;
    }

    /**
     * Given dL/dOutput, compute dL/dInput for each input.
     * Returns one gradient per input, same order as {@link #inputs}.
     */
    public abstract Tensor[] backward(Tensor gradOutput);

    /**
     * Sum grad along any dimension that was broadcast.
     */
    protected static Tensor unbroadcast(Tensor grad, Shape targetShape) {
        Shape gradShape = grad.getShape();
        if (gradShape.equals(targetShape))
            return grad.clone();

        // usually for scalar+matrix... scalar should get ALL the gradients
        if (targetShape.numel() == 1)
            return grad.sum();

        // make contiguous so we can iterate flat
        if (!grad.isContiguous())
            grad = grad.contiguous();

        int gradNdim = gradShape.dimensions();
        int targetNdim = targetShape.dimensions();

        // imagine:
        // - grad:   (3, 4)
        // - target:    (4)
        // - padded: (1, 4)
        int[] padded = new int[gradNdim];
        int off = gradNdim - targetNdim;
        Arrays.fill(padded, 0, off, 1);
        for (int i = off; i < gradNdim; i++)
            padded[i] = targetShape.dim(i - off);

        // for flat mapping in this new tensor
        int[] targetStrides = Shape.contiguousStridesFrom(padded);

        float[] out = new float[targetShape.numel()];
        float[] gradData = grad.getData();
        int gradOff = grad.getOffset();
        int gradNumel = gradShape.numel();

        int[] indices = new int[gradNdim];
        for (int flat = 0; flat < gradNumel; flat++) {
            int targetFlat = 0;
            for (int d = 0; d < gradNdim; d++) {
                if (padded[d] != 1) {
                    targetFlat += indices[d] * targetStrides[d];
                }
            }
            out[targetFlat] += gradData[gradOff + flat];

            // increment n-d index (rightmost first)
            for (int d = gradNdim - 1; d >= 0; d--) {
                if (++indices[d] < gradShape.dim(d))
                    break;
                indices[d] = 0;
            }
        }

        return TensorBuilder.builder()
                .shape(targetShape)
                .fromArray(out);
    }
}
