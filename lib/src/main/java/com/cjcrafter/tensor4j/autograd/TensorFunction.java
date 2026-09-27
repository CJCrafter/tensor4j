package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.memory.Storage;
import com.cjcrafter.tensor4j.ops.Ops;
import com.google.common.base.Preconditions;
import jdk.incubator.vector.VectorOperators;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

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

    /**
     * Returns the tensors this function was applied to, in argument order.
     */
    public Tensor[] getInputs() {
        return inputs;
    }

    /**
     * Given dL/dOutput, compute dL/dInput for each input.
     * Returns one gradient per input, same order as {@link #inputs}.
     */
    public abstract Tensor[] backward(Tensor gradOutput);


    protected static Tensor applyBinary(VectorOperators.Binary op,
                                        BiFunction<Tensor, Tensor, TensorFunction> fnFactory,
                                        Tensor a, Tensor b) {
        Shape resultShape = a.getShape().broadcastWith(b.getShape());
        Preconditions.checkArgument(resultShape != null,
                "cannot broadcast shapes %s and %s", a.getShape(), b.getShape());
        Storage out = Ops.binaryOp(op, a, b, resultShape);
        return Tensor.wrapResult(out, resultShape, fnFactory.apply(a, b));
    }

    protected static Tensor applyUnary(VectorOperators.Unary op,
                                       Function<Tensor, TensorFunction> fnFactory,
                                       Tensor a) {
        Storage out = Ops.unaryOp(op, a);
        return Tensor.wrapResult(out, a.getShape(), fnFactory.apply(a));
    }

    protected static Tensor applyUnaryCached(VectorOperators.Unary op,
                                             BiFunction<Tensor, Tensor, TensorFunction> fnFactory,
                                             Tensor a) {
        Storage out = Ops.unaryOp(op, a);
        return Tensor.wrapResultDeferred(out, a.getShape(), new Tensor[]{a},
                cached -> fnFactory.apply(a, cached));
    }

    protected static Tensor applyBinaryConst(VectorOperators.Binary op,
                                             Function<Tensor, TensorFunction> fnFactory,
                                             Tensor a, float b) {
        Storage out = Ops.binaryOpConst(op, a, b);
        return Tensor.wrapResult(out, a.getShape(), fnFactory.apply(a));
    }

    protected static Tensor applyRBinaryConst(VectorOperators.Binary op,
                                              Function<Tensor, TensorFunction> fnFactory,
                                              float a, Tensor b) {
        Storage out = Ops.rbinaryOpConst(op, a, b);
        return Tensor.wrapResult(out, b.getShape(), fnFactory.apply(b));
    }

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
        float[] gradData = grad.toFloatArray();
        int gradNumel = gradShape.numel();

        int[] indices = new int[gradNdim];
        for (int flat = 0; flat < gradNumel; flat++) {
            int targetFlat = 0;
            for (int d = 0; d < gradNdim; d++) {
                if (padded[d] != 1) {
                    targetFlat += indices[d] * targetStrides[d];
                }
            }
            out[targetFlat] += gradData[flat];

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
