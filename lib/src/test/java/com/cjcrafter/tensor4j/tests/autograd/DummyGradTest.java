package com.cjcrafter.tensor4j.tests.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DummyGradTest {

    @Test
    void testGradGoesToZero() {
        // Manual seed for reproducibility
        Tensor4j.manualSeed(42);

        for (int trial = 0; trial < 10; trial++) {
            int n = Tensor4j.getRandom().nextInt(1, 256);
            Tensor tensor = TensorBuilder.builder()
                    .requiresGrad()
                    .shape(1, n)
                    .randn();

            System.out.println("Trial " + trial);

            // In 100 steps, we should approach 0
            for (int step = 0; step < 100; step++) {
                // (1,n) @ (n,1) = (1,1) scalar
                Tensor result = tensor.matmul(tensor.transpose());
                assertEquals(new Shape(1, 1), result.getShape(), "Expected the result to be a scalar");

                System.out.println("\t" + step + ". " + result.item());

                result.backward();

                // gradient of x @ x^T w.r.t. x is 2x
                Tensor gradient = tensor.getGrad();
                Tensor scaled = gradient.mul(0.1f);
                tensor = tensor.sub(scaled);
            }

            // data should be approximately 0f
            float[] data = tensor.getData();
            for (int i = 0; i < data.length; i++) {
                assertEquals(0f, data[i], 1e-3, "Expected data[" + i + "] to be approx 0f");
            }
        }
    }
}
