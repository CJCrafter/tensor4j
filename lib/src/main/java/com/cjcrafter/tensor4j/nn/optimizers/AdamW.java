package com.cjcrafter.tensor4j.nn.optimizers;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.ops.Kernels;

import java.util.Collection;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * {@link Adam}, but with weight decay.
 *
 * <p><b>AdamW</b> was created to solve the problem of weight decay in
 * <b>Adam</b>. In order to encourage simpler models, in {@link SGD}, we might
 * add an {@code L2} regularization, which is equivalent to <i>weight decay</i>.
 * Adding an {@code L2} regularization to Adam, however, is not equivalent to
 * weight decay. The solution is AdamW, which lets you specify weight decay
 * directly into the optimizer.
 *
 * <p>AdamW with {@code weightDecay = 0} is equivalent to Adam.
 *
 * @see Adam
 * @see <a href="https://arxiv.org/pdf/1711.05101">Decoupled Weight Decay Regularization</a>
 */
public class AdamW implements Optimizer {

    public static final float DEFAULT_LR = 1e-3f;
    public static final float DEFAULT_BETA1 = 0.9f;
    public static final float DEFAULT_BETA2 = 0.999f;
    public static final float DEFAULT_WEIGHT_DECAY = 0f;
    public static final float DEFAULT_EPS = 1e-8f;

    private final Collection<Tensor> params;
    private final float lr;
    private final float beta1;
    private final float beta2;
    private final float weightDecay;
    private final float eps;

    private final Map<Tensor, Tensor> firstMoment = new IdentityHashMap<>();
    private final Map<Tensor, Tensor> secondMoment = new IdentityHashMap<>();

    private int t = 0;

    public AdamW(Collection<Tensor> params, float lr) {
        this(params, lr, DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_WEIGHT_DECAY, DEFAULT_EPS);
    }

    public AdamW(Collection<Tensor> params, float lr, float beta1, float beta2, float weightDecay, float eps) {
        this.params = params;
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.weightDecay = weightDecay;
        this.eps = eps;
    }

    @Override
    public void step() {
        if (Tensor4j.isGradEnabled())
            throw new IllegalStateException("Cannot step while grad is enabled... see Tensor4j.noGrad()");

        t++;
        final float biasCorr1 = 1f - (float) Math.pow(beta1, t);
        final float biasCorr2 = 1f - (float) Math.pow(beta2, t);

        for (Tensor param : params) {
            Tensor grad = param.getGrad();
            if (grad == null) continue;
            if (!grad.isContiguous()) grad = grad.contiguous();

            Tensor m = firstMoment.computeIfAbsent(param,
                    p -> TensorBuilder.builder().like(p).zeros());
            Tensor v = secondMoment.computeIfAbsent(param,
                    p -> TensorBuilder.builder().like(p).zeros());

            Kernels.adamw(
                    param.getData(), param.getOffset(),
                    grad.getData(), grad.getOffset(),
                    m.getData(), m.getOffset(),
                    v.getData(), v.getOffset(),
                    param.getShape().numel(),
                    lr, beta1, beta2,
                    biasCorr1, biasCorr2,
                    weightDecay, eps);
        }
    }

    @Override
    public void zeroGrad() {
        for (Tensor p : params) {
            p.zeroGrad();
        }
    }
}
