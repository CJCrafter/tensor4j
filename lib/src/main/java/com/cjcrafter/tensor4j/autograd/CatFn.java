package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;

/**
 * Backward for {@link Tensor#cat(int, Tensor...)}: splits the upstream gradient back into
 * one slice per concatenated input, along the same dimension.
 */
public class CatFn extends TensorFunction {

    private final int dim;
    private final int[] sizesAlongDim;

    /**
     * Records the backward function for concatenating {@code inputs} along {@code dim},
     * where {@code sizesAlongDim[i]} is the size of {@code inputs[i]} along that dimension.
     */
    public CatFn(int dim, int[] sizesAlongDim, Tensor... inputs) {
        super(inputs);
        this.dim = dim;
        this.sizesAlongDim = sizesAlongDim;
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        if (!gradOutput.isContiguous())
            gradOutput = gradOutput.contiguous();

        Shape outShape = gradOutput.getShape();
        int ndim = outShape.dimensions();

        int outer = 1;
        for (int d = 0; d < dim; d++) outer *= outShape.dim(d);
        int inner = 1;
        for (int d = dim + 1; d < ndim; d++) inner *= outShape.dim(d);
        int outDimStride = outShape.dim(dim) * inner;

        Tensor[] grads = new Tensor[inputs.length];
        float[] gradData = gradOutput.toFloatArray();

        int cursor = 0;
        for (int k = 0; k < inputs.length; k++) {
            int sizeK = sizesAlongDim[k];
            int[] dims = inputs[k].getShape().dims();
            float[] out = new float[outer * sizeK * inner];
            int inDimStride = sizeK * inner;
            for (int o = 0; o < outer; o++) {
                int outBase = o * outDimStride + cursor * inner;
                int inBase = o * inDimStride;
                System.arraycopy(gradData, outBase, out, inBase, sizeK * inner);
            }
            grads[k] = TensorBuilder.builder().shape(new Shape(dims)).fromArray(out);
            cursor += sizeK;
        }
        return grads;
    }
}
