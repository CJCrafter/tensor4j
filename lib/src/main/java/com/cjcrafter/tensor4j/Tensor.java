package com.cjcrafter.tensor4j;

import com.cjcrafter.tensor4j.autograd.*;
import com.cjcrafter.tensor4j.memory.HeapFloatStorage;
import com.cjcrafter.tensor4j.memory.Storage;
import com.cjcrafter.tensor4j.ops.Ops;
import com.google.common.base.Preconditions;

import java.util.*;

import static jdk.incubator.vector.VectorOperators.*;

/**
 * A multi-dimensional matrix containing 32 bit floats.
 *
 * <p>By default, data is stored contiguously in a row-major format. This means
 * that looping over the underlying {@code float[] data} is equivalent to
 * looping over the logical ordering of the tensor. However, some methods, like
 * {@link #transpose()} can change the stride of the data. This means that the
 * only "safe" way to iterate over a tensor is:
 * <pre>{@code
 *         float[] dst = new float[n];
 *         int[] indices = new int[shape.dimensions()];
 *
 *         for (int flat = 0; flat < n; flat++) {
 *             dst[flat] = data[offset + shape.flatIndex(indices)];
 *
 *             // increment nd index (rightmost first)
 *             for (int d = indices.length - 1; d >= 0; d--) {
 *                 if (++indices[d] < shape.dim(d)) break;
 *                 indices[d] = 0;
 *             }
 *         }
 *         // now dst is an array with all elements in logical order
 * }</pre>
 * This is equivalent to calling {@link #contiguous()} then grabbing the
 * underlying {@code float[] data} using {@link #getData()}.
 *
 * <p>Most tensor operations in this class are <b>const</b>, meaning that they
 * return a new tensor object with a new underlying float array. The exception
 * to this rule is {@link #transpose()} and {@link #view(int...)}, which
 * re-use the underlying float array.
 */
public class Tensor implements Cloneable {

    private final Storage storage;
    private final Shape shape;
    private final int offset;

    // autograd impl... when requiresGrade==false, these objects will be null
    private final boolean requiresGrad;
    private Tensor grad;
    private TensorFunction gradFn;

    Tensor(Storage storage, Shape shape, int offset, boolean requiresGrad) {
        Preconditions.checkNotNull(storage, "storage");
        Preconditions.checkNotNull(shape, "shape");
        this.storage = storage;
        this.shape = shape;
        this.offset = offset;
        this.requiresGrad = requiresGrad;
    }

    Tensor(float[] data, Shape shape, int offset, boolean requiresGrad) {
        this(new HeapFloatStorage(data), shape, offset, requiresGrad);
    }

    // ---- accessors ----

    public Storage getStorage() {
        return storage;
    }

    public DType dtype() {
        return storage.dtype();
    }

    public Device device() {
        return storage.device();
    }

    /**
     * Direct access to the backing float array. Only valid for heap-backed F32 storage.
     */
    public float[] getData() {
        if (!(storage instanceof HeapFloatStorage h))
            throw new UnsupportedOperationException(
                    "getData() requires heap F32 storage, got " + storage.dtype() + " on " + storage.device());
        return h.array();
    }

    private float[] data() {
        return ((HeapFloatStorage) storage).array();
    }

    public Shape getShape() {
        return shape;
    }

    public int getOffset() {
        return offset;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    public Tensor getGrad() {
        return grad;
    }

    public TensorFunction getGradFn() {
        return gradFn;
    }

    void setGradFn(TensorFunction fn) {
        this.gradFn = fn;
    }

    public void zeroGrad() {
        this.grad = null;
    }

    private void checkInPlaceSafe() {
        if (this.requiresGrad && this.gradFn != null)
            throw new IllegalStateException(
                    "in-place operation on a tensor that is part of the computation graph");
        if (offset != 0 || data().length != shape.numel() || !isContiguous())
            throw new UnsupportedOperationException(
                    "in-place operation not supported on views or non-contiguous tensors");
    }

    public void backward() {
        if (shape.numel() != 1)
            throw new IllegalStateException("backward() only supported on scalar tensors, got " + shape);

        // seed with dL/dL = 1
        this.grad = new Tensor(new float[]{1f}, new Shape(1, 1), 0, false);

        // build reverse-topological order of the forward graph so gradient is
        // fully accumulated from all paths before propagating further back.
        List<Tensor> topo = new ArrayList<>();
        Set<Tensor> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        buildTopo(this, visited, topo);

        for (int idx = topo.size() - 1; idx >= 0; idx--) {
            Tensor node = topo.get(idx);
            if (node.gradFn == null) continue;
            if (node.grad == null) continue; // unreachable from loss

            Tensor[] grads = node.gradFn.backward(node.grad);
            Tensor[] inputs = node.gradFn.getInputs();
            for (int i = 0; i < inputs.length; i++) {
                Tensor input = inputs[i];
                if (!input.requiresGrad) continue;

                if (input.grad == null) {
                    // This is a little bit dangerous since the grad function
                    // is responsible for cloning these tensors.
                    input.grad = grads[i];
                } else {
                    // accumulate in-place, bypassing checkInPlaceSafe since
                    // grad tensors are never part of the forward graph
                    Ops.binaryOp(ADD, input.grad, grads[i], input.grad.shape, input.grad.data());
                }
            }
        }
    }

    private static void buildTopo(Tensor node, Set<Tensor> visited, List<Tensor> out) {
        if (!visited.add(node)) return;
        if (node.gradFn != null) {
            for (Tensor input : node.gradFn.getInputs()) {
                buildTopo(input, visited, out);
            }
        }
        out.add(node);
    }

    public float item() {
        if (shape.numel() != 1)
            throw new IllegalStateException("item() only valid on single-element tensors, got shape " + shape);
        return data()[offset];
    }

    public float get(int... indices) {
        return data()[offset + shape.flatIndex(indices)];
    }

    public void set(float val, int... indices) {
        data()[offset + shape.flatIndex(indices)] = val;
    }

    public boolean isContiguous() {
        return shape.isContiguous();
    }

    public Tensor transpose() {
        Tensor tensor = new Tensor(storage, shape.transpose(), offset, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            tensor.gradFn = new TransposeFn(this);
        }
        return tensor;
    }

    public Tensor view(int... newDims) {
        Shape newShape = shape.reshape(newDims);
        Tensor tensor = new Tensor(storage, newShape, offset, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            tensor.gradFn = new ViewFn(this);
        }
        return tensor;
    }

    public Tensor contiguous() {
        if (isContiguous())
            return this;

        int n = shape.numel();
        float[] dst = new float[n];
        int[] indices = new int[shape.dimensions()];
        float[] src = data();

        for (int flat = 0; flat < n; flat++) {
            dst[flat] = src[offset + shape.flatIndex(indices)];

            // increment nd index (rightmost first)
            for (int d = indices.length - 1; d >= 0; d--) {
                if (++indices[d] < shape.dim(d)) break;
                indices[d] = 0;
            }
        }

        Tensor tensor = new Tensor(dst, new Shape(shape.dims()), 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            tensor.gradFn = new IdentityFn(this);
        }
        return tensor;
    }

    @Override
    public Tensor clone() {
        // TODO: allow cloning into different memory formats
        float[] src = data();
        Tensor tensor = new Tensor(Arrays.copyOf(src, src.length), shape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            tensor.gradFn = new IdentityFn(this);
        }
        return tensor;
    }

    public Tensor detach() {
        return new Tensor(storage, shape, offset, false);
    }

    public static Tensor cat(int dim, Tensor... tensors) {
        if (tensors == null || tensors.length == 0)
            throw new IllegalArgumentException("cat requires at least one tensor");

        Shape first = tensors[0].shape;
        int ndim = first.dimensions();
        if (dim < 0 || dim >= ndim)
            throw new IllegalArgumentException("dim " + dim + " out of range for " + ndim + "D tensor");

        int[] outDims = first.dims();
        int[] sizesAlongDim = new int[tensors.length];
        sizesAlongDim[0] = first.dim(dim);
        int catTotal = first.dim(dim);

        for (int i = 1; i < tensors.length; i++) {
            Shape s = tensors[i].shape;
            if (s.dimensions() != ndim)
                throw new IllegalArgumentException("cat: mismatched ndim at input " + i);
            for (int d = 0; d < ndim; d++) {
                if (d == dim) continue;
                if (s.dim(d) != first.dim(d))
                    throw new IllegalArgumentException(
                            "cat: shapes differ at dim " + d + ": "
                                    + first + " vs " + s);
            }
            sizesAlongDim[i] = s.dim(dim);
            catTotal += s.dim(dim);
        }
        outDims[dim] = catTotal;
        Shape outShape = new Shape(outDims);

        int outer = 1;
        for (int d = 0; d < dim; d++) outer *= first.dim(d);
        int inner = 1;
        for (int d = dim + 1; d < ndim; d++) inner *= first.dim(d);
        int outDimStride = catTotal * inner;

        float[] out = new float[outShape.numel()];
        boolean grad = false;
        int cursor = 0;
        for (int k = 0; k < tensors.length; k++) {
            Tensor t = tensors[k].isContiguous() ? tensors[k] : tensors[k].contiguous();
            grad |= t.requiresGrad;
            int sizeK = sizesAlongDim[k];
            int inDimStride = sizeK * inner;
            float[] tData = t.data();
            int tOff = t.offset;
            for (int o = 0; o < outer; o++) {
                int outBase = o * outDimStride + cursor * inner;
                int inBase = tOff + o * inDimStride;
                System.arraycopy(tData, inBase, out, outBase, sizeK * inner);
            }
            cursor += sizeK;
        }

        Tensor result = new Tensor(out, outShape, 0, grad);
        if (grad && Tensor4j.isGradEnabled()) {
            result.gradFn = new CatFn(dim, sizesAlongDim, tensors);
        }
        return result;
    }

    public Tensor indexSelect(int dim, int... indices) {
        if (!isContiguous())
            return contiguous().indexSelect(dim, indices);

        int ndim = shape.dimensions();
        if (dim < 0 || dim >= ndim)
            throw new IllegalArgumentException("dim " + dim + " out of range for " + ndim + "D tensor");

        int[] newDims = shape.dims();
        newDims[dim] = indices.length;

        // size of one slice along dim
        int sliceSize = 1;
        for (int d = dim + 1; d < ndim; d++)
            sliceSize *= shape.dim(d);

        // stride to jump between slices along dim
        int dimStride = shape.dim(dim) * sliceSize;

        // number of outer repetitions (product of dims before dim)
        int outerSize = 1;
        for (int d = 0; d < dim; d++)
            outerSize *= shape.dim(d);

        float[] out = new float[outerSize * indices.length * sliceSize];
        float[] src = data();
        int outPos = 0;

        for (int outer = 0; outer < outerSize; outer++) {
            int outerOff = offset + outer * dimStride;
            for (int idx : indices) {
                System.arraycopy(src, outerOff + idx * sliceSize, out, outPos, sliceSize);
                outPos += sliceSize;
            }
        }

        Tensor result = new Tensor(out, new Shape(newDims), 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new IndexSelectFn(this, dim, indices);
        }
        return result;
    }

    // -------------- //
    // --- MATMUL --- //
    // -------------- //

    public Tensor matmul(Tensor other) {
        int aDim = shape.dimensions();
        int bDim = other.shape.dimensions();

        if (aDim > 3 || bDim > 3)
            throw new IllegalArgumentException("matmul requires 2D or 3D tensors, got " + aDim + "D and " + bDim + "D");

        // inner dimensions must agree
        int k1 = shape.dim(aDim - 1);
        int k2 = other.shape.dim(bDim - 2);
        if (k1 != k2)
            throw new IllegalArgumentException("matmul inner dimensions don't match: " + shape + " @ " + other.shape);

        // compute result shape
        int m = shape.dim(aDim - 2);
        int n = other.shape.dim(bDim - 1);
        Shape resultShape;

        if (aDim == 2 && bDim == 2) {
            resultShape = new Shape(m, n);
        } else {
            // 1 must be 3d... batch dimension
            int batch;
            if (aDim == 3 && bDim == 3) {
                if (shape.dim(0) != other.shape.dim(0))
                    throw new IllegalArgumentException("batch dimensions don't match: " + shape + " @ " + other.shape);
                batch = shape.dim(0);
            } else {
                batch = aDim == 3 ? shape.dim(0) : other.shape.dim(0);
            }
            resultShape = new Shape(batch, m, n);
        }

        float[] resultData = Ops.matmul(this, other, resultShape);
        boolean grad = this.requiresGrad || other.requiresGrad;
        Tensor result = new Tensor(resultData, resultShape, 0, grad);
        if (grad && Tensor4j.isGradEnabled()) {
            result.gradFn = new MatmulFn(this, other);
        }
        return result;
    }

    // ------------------ //
    // --- REDUCTIONS --- //
    // ------------------ //

    public Tensor sum() {
        float val = Ops.sum(this);
        Tensor result = new Tensor(new float[]{val}, new Shape(1, 1), 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new SumFn(this);
        }
        return result;
    }

    public Tensor sum(int dim) {
        if (dim < 0 || dim >= shape.dimensions())
            throw new IllegalArgumentException("dim " + dim + " out of range for " + shape.dimensions() + "D tensor");

        float[] resultData = Ops.sumDim(this, dim);

        // result shape: same as input but with dims[dim] = 1
        int[] newDims = shape.dims();
        newDims[dim] = 1;
        Shape resultShape = new Shape(newDims);

        Tensor result = new Tensor(resultData, resultShape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new SumDimFn(this, dim);
        }
        return result;
    }

    public Tensor max(int dim) {
        if (dim < 0 || dim >= shape.dimensions())
            throw new IllegalArgumentException("dim " + dim + " out of range for " + shape.dimensions() + "D tensor");

        float[] resultData = Ops.maxDim(this, dim);

        int[] newDims = shape.dims();
        newDims[dim] = 1;
        Shape resultShape = new Shape(newDims);

        Tensor result = new Tensor(resultData, resultShape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new MaxDimFn(this, result, dim);
        }
        return result;
    }

    public Tensor argmax(int dim) {
        if (dim < 0 || dim >= shape.dimensions())
            throw new IllegalArgumentException("dim " + dim + " out of range for " + shape.dimensions() + "D tensor");

        float[] resultData = Ops.argmaxDim(this, dim);

        int[] newDims = shape.dims();
        newDims[dim] = 1;
        return new Tensor(resultData, new Shape(newDims), 0, false);
    }

    public Tensor logSoftmax(int dim) {
        if (dim < 0 || dim >= shape.dimensions())
            throw new IllegalArgumentException("dim " + dim + " out of range for " + shape.dimensions() + "D tensor");

        Tensor maxes = this.max(dim);
        Tensor shifted = this.sub(maxes);
        Tensor sumExp = shifted.exp().sum(dim);
        Tensor logSumExp = sumExp.log();
        return shifted.sub(logSumExp);
    }

    public Tensor softmax(int dim) {
        if (dim < 0 || dim >= shape.dimensions())
            throw new IllegalArgumentException("dim " + dim + " out of range for " + shape.dimensions() + "D tensor");

        Tensor shifted = this.sub(this.max(dim).detach());
        Tensor expd = shifted.exp();
        return expd.div(expd.sum(dim));
    }

    public Tensor maskedSoftmax(Tensor mask, int dim) {
        // https://mcognetta.github.io/posts/masked_softmax/
        if (dim < 0 || dim >= shape.dimensions())
            throw new IllegalArgumentException("dim " + dim + " out of range for " + shape.dimensions() + "D tensor");

        Tensor m = mask.requiresGrad() ? mask.detach() : mask;
        return this.add(m.rsub(1f).mul(-1e30f)).softmax(dim);
    }

    public Tensor mean() {
        float val = Ops.mean(this);
        Tensor result = new Tensor(new float[]{val}, new Shape(1, 1), 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new MeanFn(this);
        }
        return result;
    }

    // -------------------------- //
    // --- UNARY ELEMENT-WISE --- //
    // -------------------------- //


    public Tensor tanh() {
        float[] resultData = new float[shape.numel()];
        Ops.unaryOp(TANH, this, resultData);
        Tensor result = new Tensor(resultData, shape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new TanhFn(this, result);
        }
        return result;
    }

    public Tensor abs() {
        float[] resultData = new float[shape.numel()];
        Ops.unaryOp(ABS, this, resultData);
        Tensor result = new Tensor(resultData, shape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new AbsFn(this);
        }
        return result;
    }

    public Tensor abs_() {
        checkInPlaceSafe();
        Ops.unaryOp(ABS, this, data());
        return this;
    }

    public Tensor exp() {
        float[] resultData = new float[shape.numel()];
        Ops.unaryOp(EXP, this, resultData);
        Tensor result = new Tensor(resultData, shape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new ExpFn(this, result);
        }
        return result;
    }

    public Tensor exp_() {
        checkInPlaceSafe();
        Ops.unaryOp(EXP, this, data());
        return this;
    }

    public Tensor log() {
        float[] resultData = new float[shape.numel()];
        Ops.unaryOp(LOG, this, resultData);
        Tensor result = new Tensor(resultData, shape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new LogFn(this);
        }
        return result;
    }

    public Tensor log_() {
        checkInPlaceSafe();
        Ops.unaryOp(LOG, this, data());
        return this;
    }

    public Tensor sqrt() {
        float[] resultData = new float[shape.numel()];
        Ops.unaryOp(SQRT, this, resultData);
        Tensor result = new Tensor(resultData, shape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new SqrtFn(this, result);
        }
        return result;
    }

    public Tensor sqrt_() {
        checkInPlaceSafe();
        Ops.unaryOp(SQRT, this, data());
        return this;
    }

    public Tensor square() {
        // TODO: make optimized kernel
        float[] resultData = new float[shape.numel()];
        Ops.binaryOp(MUL, this, this, shape, resultData);
        Tensor result = new Tensor(resultData, shape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new SquareFn(this);
        }
        return result;
    }

    public Tensor square_() {
        checkInPlaceSafe();
        Ops.binaryOp(MUL, this, this, shape, data());
        return this;
    }

    // --------------------------- //
    // --- BINARY ELEMENT-WISE --- //
    // --------------------------- //

    public Tensor add(Tensor other) {
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (resultShape == null)
            throw new IllegalArgumentException("cannot broadcast shapes " + this.shape + " and " + other.shape);

        float[] resultData = new float[resultShape.numel()];
        Ops.binaryOp(ADD, this, other, resultShape, resultData);
        boolean grad = this.requiresGrad || other.requiresGrad;
        Tensor result = new Tensor(resultData, resultShape, 0, grad);
        if (grad && Tensor4j.isGradEnabled()) {
            result.gradFn = new AddFn(this, other);
        }
        return result;
    }

    public Tensor add_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (!resultShape.equals(this.shape))
            throw new IllegalArgumentException("in-place: result shape must equal this.shape");
        Ops.binaryOp(ADD, this, other, resultShape, data());
        return this;
    }

    public Tensor add(float other) {
        float[] resultData = new float[shape.numel()];
        Ops.binaryOpConst(ADD, this, other, resultData);
        Tensor result = new Tensor(resultData, shape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new AddConstFn(this, other);
        }
        return result;
    }

    public Tensor add_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(ADD, this, other, data());
        return this;
    }

    public Tensor sub(Tensor other) {
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (resultShape == null)
            throw new IllegalArgumentException(
                    "cannot broadcast shapes " + this.shape + " and " + other.shape);
        float[] resultData = new float[resultShape.numel()];
        Ops.binaryOp(SUB, this, other, resultShape, resultData);
        boolean grad = this.requiresGrad || other.requiresGrad;
        Tensor result = new Tensor(resultData, resultShape, 0, grad);
        if (grad && Tensor4j.isGradEnabled()) {
            result.gradFn = new SubFn(this, other);
        }
        return result;
    }

    public Tensor sub_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (!resultShape.equals(this.shape))
            throw new IllegalArgumentException("in-place: result shape must equal this.shape");
        Ops.binaryOp(SUB, this, other, resultShape, data());
        return this;
    }

    public Tensor sub(float other) {
        float[] resultData = new float[shape.numel()];
        Ops.binaryOpConst(SUB, this, other, resultData);
        Tensor result = new Tensor(resultData, shape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new AddConstFn(this, other);
        }
        return result;
    }

    public Tensor sub_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(SUB, this, other, data());
        return this;
    }

    public Tensor rsub(float other) {
        float[] resultData = new float[shape.numel()];
        Ops.rbinaryOpConst(SUB, other, this, resultData);
        Tensor result = new Tensor(resultData, shape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new SubRConstFn(other, this);
        }
        return result;
    }

    public Tensor rsub_(float other) {
        checkInPlaceSafe();
        Ops.rbinaryOpConst(SUB, other, this, data());
        return this;
    }

    public Tensor mul(Tensor other) {
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (resultShape == null)
            throw new IllegalArgumentException(
                    "cannot broadcast shapes " + this.shape + " and " + other.shape);
        float[] resultData = new float[resultShape.numel()];
        Ops.binaryOp(MUL, this, other, resultShape, resultData);
        boolean grad = this.requiresGrad || other.requiresGrad;
        Tensor result = new Tensor(resultData, resultShape, 0, grad);
        if (grad && Tensor4j.isGradEnabled()) {
            result.gradFn = new MulFn(this, other);
        }
        return result;
    }

    public Tensor mul_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (!resultShape.equals(this.shape))
            throw new IllegalArgumentException("in-place: result shape must equal this.shape");
        Ops.binaryOp(MUL, this, other, resultShape, data());
        return this;
    }

    public Tensor mul(float other) {
        float[] resultData = new float[shape.numel()];
        Ops.binaryOpConst(MUL, this, other, resultData);
        Tensor result = new Tensor(resultData, shape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new MulConstFn(this, other);
        }
        return result;
    }

    public Tensor mul_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(MUL, this, other, data());
        return this;
    }

    public Tensor div(Tensor other) {
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (resultShape == null)
            throw new IllegalArgumentException(
                    "cannot broadcast shapes " + this.shape + " and " + other.shape);
        float[] resultData = new float[resultShape.numel()];
        Ops.binaryOp(DIV, this, other, resultShape, resultData);
        boolean grad = this.requiresGrad || other.requiresGrad;
        Tensor result = new Tensor(resultData, resultShape, 0, grad);
        if (grad && Tensor4j.isGradEnabled()) {
            result.gradFn = new DivFn(this, other);
        }
        return result;
    }

    public Tensor div_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (!resultShape.equals(this.shape))
            throw new IllegalArgumentException("in-place: result shape must equal this.shape");
        Ops.binaryOp(DIV, this, other, resultShape, data());
        return this;
    }

    public Tensor div(float other) {
        float[] resultData = new float[shape.numel()];
        Ops.binaryOpConst(DIV, this, other, resultData);
        Tensor result = new Tensor(resultData, shape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new DivConstFn(this, other);
        }
        return result;
    }

    public Tensor div_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(DIV, this, other, data());
        return this;
    }

    public Tensor rdiv(float other) {
        float[] resultData = new float[shape.numel()];
        Ops.rbinaryOpConst(DIV, other, this, resultData);
        Tensor result = new Tensor(resultData, shape, 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            result.gradFn = new DivRConstFn(other, this);
        }
        return result;
    }

    public Tensor rdiv_(float other) {
        checkInPlaceSafe();
        Ops.rbinaryOpConst(DIV, other, this, data());
        return this;
    }

    public Tensor max(Tensor other) {
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (resultShape == null)
            throw new IllegalArgumentException("cannot broadcast shapes " + this.shape + " and " + other.shape);
        float[] resultData = new float[resultShape.numel()];
        Ops.binaryOp(MAX, this, other, resultShape, resultData);
        boolean grad = this.requiresGrad || other.requiresGrad;
        Tensor result = new Tensor(resultData, resultShape, 0, grad);
        if (grad && Tensor4j.isGradEnabled()) {
            result.gradFn = new MaxFn(this, other);
        }
        return result;
    }

    public Tensor max_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (!resultShape.equals(this.shape))
            throw new IllegalArgumentException("in-place: result shape must equal this.shape");
        Ops.binaryOp(MAX, this, other, resultShape, data());
        return this;
    }

    public Tensor max(float other) {
        float[] resultData = new float[shape.numel()];
        Ops.binaryOpConst(MAX, this, other, resultData);
        // relu is a special case where we definitely don't need to change the order
        // at all or make it contiguous... we just need the shape to match exactly.
        Tensor tensor = new Tensor(resultData, new Shape(shape.dims(), shape.strides()), 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            tensor.gradFn = new MaxConstFn(this, other);
        }
        return tensor;
    }

    public Tensor max_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(MAX, this, other, data());
        return this;
    }

    public Tensor min(Tensor other) {
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (resultShape == null)
            throw new IllegalArgumentException("cannot broadcast shapes " + this.shape + " and " + other.shape);
        float[] resultData = new float[resultShape.numel()];
        Ops.binaryOp(MIN, this, other, resultShape, resultData);
        boolean grad = this.requiresGrad || other.requiresGrad;
        Tensor result = new Tensor(resultData, resultShape, 0, grad);
        if (grad && Tensor4j.isGradEnabled()) {
            result.gradFn = new MinFn(this, other);
        }
        return result;
    }

    public Tensor min_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (!resultShape.equals(this.shape))
            throw new IllegalArgumentException("in-place: result shape must equal this.shape");
        Ops.binaryOp(MIN, this, other, resultShape, data());
        return this;
    }

    public Tensor min(float other) {
        float[] resultData = new float[shape.numel()];
        Ops.binaryOpConst(MIN, this, other, resultData);
        Tensor tensor = new Tensor(resultData, new Shape(shape.dims(), shape.strides()), 0, requiresGrad);
        if (requiresGrad && Tensor4j.isGradEnabled()) {
            tensor.gradFn = new MinConstFn(this, other);
        }
        return tensor;
    }

    public Tensor min_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(MIN, this, other, data());
        return this;
    }

    // ------------------------------- //
    // --- COMPARISON ELEMENT-WISE --- //
    // ------------------------------- //

    public Tensor eq(Tensor other) {
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (resultShape == null)
            throw new IllegalArgumentException("cannot broadcast shapes " + this.shape + " and " + other.shape);
        float[] resultData = new float[resultShape.numel()];
        Ops.comparisonOp(EQ, this, other, resultShape, resultData);
        // always no grad... gradient is strictly 0
        return new Tensor(resultData, resultShape, 0, false);
    }

    public Tensor eq_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (!this.shape.equals(resultShape))
            throw new IllegalArgumentException("cannot broadcast shapes " + this.shape + " and " + other.shape);
        Ops.comparisonOp(EQ, this, other, resultShape, data());
        return this;
    }

    public Tensor eq(float other) {
        float[] resultData = new float[shape.numel()];
        Ops.comparisonOpConst(EQ, this, other, resultData);
        // always no grad... gradient is strictly 0
        return new Tensor(resultData, shape, 0, false);
    }

    public Tensor eq_(float other) {
        checkInPlaceSafe();
        Ops.comparisonOpConst(EQ, this, other, data());
        // always no grad... gradient is strictly 0
        return this;
    }

    public Tensor lt(Tensor other) {
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (resultShape == null)
            throw new IllegalArgumentException("cannot broadcast shapes " + this.shape + " and " + other.shape);
        float[] resultData = new float[resultShape.numel()];
        Ops.comparisonOp(LT, this, other, resultShape, resultData);
        // always no grad... gradient is strictly 0
        return new Tensor(resultData, resultShape, 0, false);
    }

    public Tensor lt_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (!this.shape.equals(resultShape))
            throw new IllegalArgumentException("cannot broadcast shapes " + this.shape + " and " + other.shape);
        Ops.comparisonOp(LT, this, other, resultShape, data());
        // always no grad... gradient is strictly 0
        return this;
    }

    public Tensor lt(float other) {
        float[] resultData = new float[shape.numel()];
        Ops.comparisonOpConst(LT, this, other, resultData);
        // always no grad... gradient is strictly 0
        return new Tensor(resultData, shape, 0, false);
    }

    public Tensor lt_(float other) {
        checkInPlaceSafe();
        Ops.comparisonOpConst(LT, this, other, data());
        // always no grad... gradient is strictly 0
        return this;
    }

    public Tensor le(Tensor other) {
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (resultShape == null)
            throw new IllegalArgumentException("cannot broadcast shapes " + this.shape + " and " + other.shape);
        float[] resultData = new float[resultShape.numel()];
        Ops.comparisonOp(LE, this, other, resultShape, resultData);
        // always no grad... gradient is strictly 0
        return new Tensor(resultData, resultShape, 0, false);
    }

    public Tensor le_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (!this.shape.equals(resultShape))
            throw new IllegalArgumentException("cannot broadcast shapes " + this.shape + " and " + other.shape);
        Ops.comparisonOp(LE, this, other, resultShape, data());
        // always no grad... gradient is strictly 0
        return this;
    }

    public Tensor le(float other) {
        float[] resultData = new float[shape.numel()];
        Ops.comparisonOpConst(LE, this, other, resultData);
        // always no grad... gradient is strictly 0
        return new Tensor(resultData, shape, 0, false);
    }

    public Tensor le_(float other) {
        checkInPlaceSafe();
        Ops.comparisonOpConst(LE, this, other, data());
        // always no grad... gradient is strictly 0
        return this;
    }

    public Tensor gt(Tensor other) {
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (resultShape == null)
            throw new IllegalArgumentException("cannot broadcast shapes " + this.shape + " and " + other.shape);
        float[] resultData = new float[resultShape.numel()];
        Ops.comparisonOp(GT, this, other, resultShape, resultData);
        // always no grad... gradient is strictly 0
        return new Tensor(resultData, resultShape, 0, false);
    }

    public Tensor gt_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (this.shape.equals(resultShape))
            throw new IllegalArgumentException("cannot broadcast shapes " + this.shape + " and " + other.shape);
        Ops.comparisonOp(GT, this, other, resultShape, data());
        // always no grad... gradient is strictly 0
        return this;
    }

    public Tensor gt(float other) {
        float[] resultData = new float[shape.numel()];
        Ops.comparisonOpConst(GT, this, other, resultData);
        // always no grad... gradient is strictly 0
        return new Tensor(resultData, shape, 0, false);
    }

    public Tensor gt_(float other) {
        Ops.comparisonOpConst(GT, this, other, data());
        // always no grad... gradient is strictly 0
        return this;
    }

    public Tensor ge(Tensor other) {
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (resultShape == null)
            throw new IllegalArgumentException("cannot broadcast shapes " + this.shape + " and " + other.shape);
        float[] resultData = new float[resultShape.numel()];
        Ops.comparisonOp(GE, this, other, resultShape, resultData);
        // always no grad... gradient is strictly 0
        return new Tensor(resultData, resultShape, 0, false);
    }

    public Tensor ge_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        if (!this.shape.equals(resultShape))
            throw new IllegalArgumentException("cannot broadcast shapes " + this.shape + " and " + other.shape);
        Ops.comparisonOp(GE, this, other, resultShape, data());
        // always no grad... gradient is strictly 0
        return this;
    }

    public Tensor ge(float other) {
        float[] resultData = new float[shape.numel()];
        Ops.comparisonOpConst(GE, this, other, resultData);
        // always no grad... gradient is strictly 0
        return new Tensor(resultData, shape, 0, false);
    }

    public Tensor ge_(float other) {
        checkInPlaceSafe();
        Ops.comparisonOpConst(GE, this, other, data());
        // always no grad... gradient is strictly 0
        return this;
    }

    // ---------------------------- //
    // --- TERNARY ELEMENT-WISE --- //
    // ---------------------------- //

    //public Tensor fma(Tensor b, Tensor c) {
    //    Shape resultShape = this.shape.broadcastWith(b.shape);
    //    if (resultShape == null)
    //        throw new IllegalArgumentException("cannot broadcast shapes " + this.shape + " and " + b.shape);
    //    float[] resultData = new float[resultShape.numel()];
    //    Ops.ternaryOp(FMA, this, b, c, resultShape, resultData);
    //    // always no grad... gradient is strictly 0
    //    return new Tensor(resultData, resultShape, 0, false);
    //}

    // --------------------------- //
    // --- MISC / JAVA METHODS --- //
    // --------------------------- //

    @Override
    public String toString() {
        int ndim = shape.dimensions();
        int numel = shape.numel();
        float[] src = data();

        // collect all values for consistent formatting
        float[] values = new float[numel];
        int[] indices = new int[ndim];
        for (int i = 0; i < numel; i++) {
            values[i] = src[offset + shape.flatIndex(indices)];
            for (int d = ndim - 1; d >= 0; d--) {
                if (++indices[d] < shape.dim(d)) break;
                indices[d] = 0;
            }
        }

        // pytorch will simplify floats to integers, and use scientific
        // notation for extreme values.
        boolean allInt = true;
        boolean needSci = false;
        for (float v : values) {
            if (v != (int) v) allInt = false;
            if (v != 0f && (Math.abs(v) >= 1e8 || Math.abs(v) < 1e-4))
                needSci = true;
        }

        // format all values and find max width for alignment
        String[] formatted = new String[numel];
        for (int i = 0; i < numel; i++) {
            if (needSci)
                formatted[i] = String.format("%e", values[i]);
            else if (allInt)
                formatted[i] = String.format("%.0f.", values[i]);
            else
                formatted[i] = String.format("%.4f", values[i]);
        }

        int maxWidth = 0;
        for (String s : formatted)
            maxWidth = Math.max(maxWidth, s.length());

        // pad all values to max width (right-aligned)
        for (int i = 0; i < numel; i++)
            formatted[i] = " ".repeat(maxWidth - formatted[i].length()) + formatted[i];

        // build the nested structure recursively
        StringBuilder sb = new StringBuilder("tensor(");
        int[] pos = {0};
        appendDim(sb, formatted, 0, ndim, "       ", pos);
        sb.append(")");
        return sb.toString();
    }

    private void appendDim(StringBuilder sb, String[] formatted, int dim, int ndim, String indent, int[] pos) {
        sb.append("[");
        int size = shape.dim(dim);

        if (dim == ndim - 1) {
            // innermost dimension: print values
            for (int i = 0; i < size; i++) {
                if (i > 0) sb.append(", ");
                sb.append(formatted[pos[0]++]);
            }
        } else {
            for (int i = 0; i < size; i++) {
                if (i > 0) {
                    sb.append(",");
                    // blank lines between slices for 3D+
                    int blankLines = ndim - dim - 2;
                    for (int b = 0; b <= blankLines; b++) sb.append("\n");
                    sb.append(indent).append(" ".repeat(dim + 1));
                }
                appendDim(sb, formatted, dim + 1, ndim, indent, pos);
            }
        }
        sb.append("]");
    }

}
