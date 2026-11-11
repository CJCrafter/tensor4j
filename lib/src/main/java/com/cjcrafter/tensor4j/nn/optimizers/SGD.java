package com.cjcrafter.tensor4j.nn.optimizers;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;

import com.google.common.base.Preconditions;

import java.util.Collection;

/**
 * Plain stochastic gradient descent: {@code param -= lr * grad}.
 */
public class SGD implements Optimizer {

    private final Collection<Tensor> params;
    private float lr;

    /**
     * Creates an SGD optimizer over {@code params} with learning rate {@code lr}.
     * @param params the parameters to update
     * @param lr the learning rate
     */
    public SGD(Collection<Tensor> params, float lr) {
        this.params = params;
        this.lr = lr;
    }

    @Override
    public void step() {
        Preconditions.checkState(!Tensor4j.isGradEnabled(),
                "Cannot step while grad is enabled... see Tensor4j.noGrad()");

        for (Tensor param : params) {
            Tensor grad = param.getGrad();
            if (grad == null)
                continue;

            grad.mul_(lr);
            param.sub_(grad);
        }
    }

    @Override
    public void setLearningRate(float lr) {
        this.lr = lr;
    }

    @Override
    public void zeroGrad() {
        for (Tensor p : params) {
            p.zeroGrad();
        }
    }
}
