package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;

/**
 * Backward for {@code indexSelect(dim, indices)}.
 */
public class IndexSelectFn extends TensorFunction {

    private final int dim;
    private final int[] indices;

    public IndexSelectFn(Tensor input, int dim, int[] indices) {
        super(input);
        this.dim = dim;
        this.indices = indices;
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor input = inputs[0];
        Shape inputShape = input.getShape();
        int ndim = inputShape.dimensions();

        if (!gradOutput.isContiguous())
            gradOutput = gradOutput.contiguous();

        // size of one slice along dim
        int sliceSize = 1;
        for (int d = dim + 1; d < ndim; d++)
            sliceSize *= inputShape.dim(d);

        int outerSize = 1;
        for (int d = 0; d < dim; d++)
            outerSize *= inputShape.dim(d);

        int inputDimStride = inputShape.dim(dim) * sliceSize;
        int gradDimStride = indices.length * sliceSize;

        float[] out = new float[inputShape.numel()];
        float[] gradData = gradOutput.getData();
        int gradOff = gradOutput.getOffset();

        for (int outer = 0; outer < outerSize; outer++) {
            int outOuter = outer * inputDimStride;
            int gradOuter = gradOff + outer * gradDimStride;
            for (int k = 0; k < indices.length; k++) {
                int dstBase = outOuter + indices[k] * sliceSize;
                int srcBase = gradOuter + k * sliceSize;
                for (int i = 0; i < sliceSize; i++) {
                    out[dstBase + i] += gradData[srcBase + i];
                }
            }
        }

        return new Tensor[]{
            TensorBuilder.builder().shape(inputShape).fromArray(out)
        };
    }
}
