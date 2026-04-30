package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.nn.layers.Linear;
import com.cjcrafter.tensor4j.nn.layers.Module;
import com.cjcrafter.tensor4j.nn.layers.Sequential;
import com.cjcrafter.tensor4j.nn.layers.Tanh;
import com.cjcrafter.tensor4j.nn.losses.Loss;
import com.cjcrafter.tensor4j.nn.losses.MSELoss;
import com.cjcrafter.tensor4j.nn.optimizers.Adam;
import com.cjcrafter.tensor4j.nn.optimizers.AdamW;
import com.cjcrafter.tensor4j.nn.optimizers.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.Iterator;

import static org.junit.jupiter.api.Assertions.*;

public class AdamTest {

    private static final float EPSILON = 1e-6f;

    @Test
    void testAdamAndAdamWEquivalent() {
        // when weight decay in AdamW is 0, then it should be the same as Adam

        Tensor4j.manualSeed(7L);
        Module model = new Sequential(
                new Linear(16, 16),
                new Tanh(),
                new Linear(16, 16),
                new Tanh(),
                new Linear(16, 1)
        );

        Tensor4j.manualSeed(7L);
        Module modelw = new Sequential(
                new Linear(16, 16),
                new Tanh(),
                new Linear(16, 16),
                new Tanh(),
                new Linear(16, 1)
        );

        float lr = Adam.DEFAULT_LR;
        Optimizer adam = new Adam(model.parameters(), lr);
        Optimizer adamw = new AdamW(modelw.parameters(), lr);
        Loss loss = new MSELoss();

        for (int i = 0; i < 150; i++) {
            adam.zeroGrad();
            adamw.zeroGrad();

            long seed = Tensor4j.getRandom().nextLong() ^ i;
            Tensor4j.manualSeed(seed);
            Tensor x = Tensor4j.builder()
                    .shape(256, 16)
                    .randn();
            Tensor y = model.forward(x);

            Tensor4j.manualSeed(seed);
            Tensor xw = Tensor4j.builder()
                    .shape(256, 16)
                    .randn();
            Tensor yw = modelw.forward(xw);

            Tensor gt = Tensor4j.builder().like(y).randn();
            Tensor lossScalar = loss.forward(y, gt);
            Tensor lossScalarW = loss.forward(yw, gt);
            assertEquals(lossScalar.item(), lossScalarW.item(), EPSILON, "Loss functions diverged");

            try (var ignored = Tensor4j.noGrad()) {
                lossScalar.backward();
                lossScalarW.backward();

                adam.step();
                adamw.step();
            }

            // parameters should be equivalent
            Iterator<Tensor> parameters = model.parameters().iterator();
            Iterator<Tensor> parametersW = modelw.parameters().iterator();

            while (parameters.hasNext() && parametersW.hasNext()) {
                Tensor parameter = parameters.next();
                Tensor parameterW = parametersW.next();

                try (var ignored = Tensor4j.noGrad()) {
                    float sum = parameter
                            .sub(parameterW)
                            .abs_()
                            .gt_(EPSILON)
                            .sum()
                            .item();
                    assertTrue(sum < EPSILON, i + ". Adam and AdamW parameters diverged");
                }
            }

            assertFalse(parameters.hasNext(), "Adam had more parameters");
            assertFalse(parametersW.hasNext(), "Adam had fewer parameters");
        }
    }
}
