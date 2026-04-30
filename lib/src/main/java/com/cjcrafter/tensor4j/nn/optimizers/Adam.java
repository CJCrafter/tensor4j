package com.cjcrafter.tensor4j.nn.optimizers;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;

import java.util.Collection;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Gradient based optimization based on adaptive estimates of lower-order
 * moments.
 *
 * <p>Generally speaking, you should prefer {@link AdamW} over Adam.
 *
 * @see AdamW
 * @see <a href="https://arxiv.org/abs/1412.6980">Adam: A Method for Stochastic Optimization</a>
 */
public class Adam implements Optimizer {

    public static final float DEFAULT_LR = 1e-3f;
    public static final float DEFAULT_BETA1 = 0.9f;
    public static final float DEFAULT_BETA2 = 0.999f;
    public static final float DEFAULT_EPS = 1e-8f;

    private final Collection<Tensor> params;
    private final float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;

    private final Map<Tensor, Tensor> firstMoment = new IdentityHashMap<>();
    private final Map<Tensor, Tensor> secondMoment = new IdentityHashMap<>();

    private int t = 0;

    public Adam(Collection<Tensor> params, float lr) {
        this(params, lr, DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPS);
    }

    public Adam(Collection<Tensor> params, float lr, float beta1, float beta2, float eps) {
        this.params = params;
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
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

            float[] pData = param.getData();
            int pOff = param.getOffset();
            float[] gData = grad.getData();
            int gOff = grad.getOffset();
            float[] mData = m.getData();
            int mOff = m.getOffset();
            float[] vData = v.getData();
            int vOff = v.getOffset();

            int n = param.getShape().numel();
            for (int i = 0; i < n; i++) {
                float g = gData[gOff + i];

                float mi = beta1 * mData[mOff + i] + (1f - beta1) * g;
                float vi = beta2 * vData[vOff + i] + (1f - beta2) * g * g;
                mData[mOff + i] = mi;
                vData[vOff + i] = vi;

                float mHat = mi / biasCorr1;
                float vHat = vi / biasCorr2;
                pData[pOff + i] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
            }
        }
    }

    @Override
    public void zeroGrad() {
        for (Tensor p : params) {
            p.zeroGrad();
        }
    }
}
