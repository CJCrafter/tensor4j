package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;

/**
 * During training, randomly zeroes some of the elements of the input tensor
 * with probability {@code p}. The zeroed elements are chosen independently for
 * each forward call, and are sampled from a Bernoulli distribution.
 *
 * <p>During testing/validation, you should always disable dropout. This can be
 * accomplished by calling {@link #eval()} on your model. Just make sure to set
 * it back to {@link #train()} before continuing.
 *
 * @see <a href="https://arxiv.org/abs/1207.0580">Improving neural networks by preventing co-adaptation of feature vectors</a>
 */
public class Dropout extends Module {

    private final float p;
    private boolean isTrain;

    public Dropout(float p) {
        this.p = p;
        this.isTrain = true;
    }

    @Override
    public Tensor forward(Tensor input) {
        if (isTrain) {
            // avoid division by 0
            if (p >= 1f)
                return input.mul(0f);

            Tensor bernoulli = Tensor4j.builder()
                    .like(input)
                    .bernoulli(p);
            return input.mul(bernoulli.mul_(1f / (1f - p)));
        } else {
            // identity function in eval mode
            return input;
        }
    }

    @Override
    public void train(boolean mode) {
        this.isTrain = mode;
    }
}
