package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.nn.layers.Linear;
import com.cjcrafter.tensor4j.nn.layers.ReLU;
import com.cjcrafter.tensor4j.nn.layers.Sequential;
import com.cjcrafter.tensor4j.nn.losses.MSELoss;
import com.cjcrafter.tensor4j.nn.optimizers.Adam;
import com.cjcrafter.tensor4j.nn.optimizers.AdamW;
import com.cjcrafter.tensor4j.nn.optimizers.Optimizer;
import com.cjcrafter.tensor4j.nn.optimizers.SGD;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Collection;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.fail;

public class XorTest {

    @FunctionalInterface
    public interface OptimizerProvider {
        Optimizer create(Collection<Tensor> params);
    }

    private static Stream<Arguments> testXorNetwork() {
        // All these optimizers should have no problem learning XOR
        return Stream.of(
                Arguments.of("SGD", (OptimizerProvider) params -> new SGD(params,0.01f)),
                Arguments.of("Adam", (OptimizerProvider) params -> new Adam(params, Adam.DEFAULT_LR)),
                Arguments.of("AdamW", (OptimizerProvider) params -> new AdamW(params, AdamW.DEFAULT_LR))
        );
    }

    @DisplayName("XOR Optimization Tests")
    @ParameterizedTest(name = "{0} optimizing XOR")
    @MethodSource
    void testXorNetwork(String optimizerName, OptimizerProvider optimizerProvider) {
        Tensor4j.manualSeed(1337);

        Tensor inputs = TensorBuilder.builder()
                .shape(4, 2)
                .fromArray(new float[]{
                        0, 0,
                        0, 1,
                        1, 0,
                        1, 1
                });
        Tensor gt = TensorBuilder.builder()
                .shape(4, 1)
                .fromArray(new float[]{0, 1, 1, 0});

        Sequential model = new Sequential(
                new Linear(2, 8),
                new ReLU(),
                new Linear(8, 1)
        );
        Optimizer optimizer = optimizerProvider.create(model.parameters());
        MSELoss loss = new MSELoss();

        for (int i = 0; i < 1000; i++) {
            optimizer.zeroGrad();
            Tensor pred = model.forward(inputs);

            boolean allMatch = true;
            for (int j = 0; j < 4; j++) {
                if (Math.round(pred.get(j)) != gt.get(j)) {
                    allMatch = false;
                    break;
                }
            }
            if (allMatch) {
                System.out.println(optimizerName + " solved XOR in " + i + " iterations");
                return;
            }

            Tensor l = loss.forward(pred, gt);
            try (var ignored = Tensor4j.noGrad()) {
                l.backward();
                optimizer.step();
            }
        }

        fail(optimizerName + " could not solve XOR");
    }
}
