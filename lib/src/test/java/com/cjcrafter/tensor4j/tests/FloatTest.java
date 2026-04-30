package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class FloatTest {

    @Test
    void assertMeanIsMoreAccurate() {
        Tensor4j.manualSeed(67);

        int trials = 1000;
        int rows = 500, cols = 500;
        int n = rows * cols;
        float mean = 0f;
        double sumErrorTotal = 0.0;
        double meanErrorTotal = 0.0;
        float worstSumError = 0f;
        float worstMeanError = 0f;

        for (int i = 0; i < trials; i++) {
            try (var ignore = Tensor4j.noGrad()) {

                // We intentionally sample from a distribution with a large mean
                // so sum/n accumulates a huge intermediate sum, stressing float32.
                Tensor bigAssMatrix = Tensor4j.builder()
                        .shape(rows, cols)
                        .randn(Tensor4j.getRandom(), mean, 1f);

                // Ground truth: compute the mean in double precision
                float[] data = bigAssMatrix.getData();
                double trueMean = 0.0;
                for (float v : data) trueMean += v;
                trueMean /= n;

                float sumResult = bigAssMatrix.sum().item() / n;
                float meanResult = bigAssMatrix.mean().item();

                float sumError = Math.abs(sumResult - (float)trueMean);
                float meanError = Math.abs(meanResult - (float)trueMean);

                sumErrorTotal += sumError;
                meanErrorTotal += meanError;
                worstSumError = Math.max(worstSumError, sumError);
                worstMeanError = Math.max(worstMeanError, meanError);
            }
        }

        double avgSumError = sumErrorTotal / trials;
        double avgMeanError = meanErrorTotal / trials;
        System.out.println("avg error:");
        System.out.println("\tsum/n = " + avgMeanError);
        System.out.println("\tfma   = " + avgMeanError);

        System.out.println("worst error:");
        System.out.println("\tsum/n = " + worstSumError);
        System.out.println("\tfma   = " + worstMeanError);

        System.out.println();
        System.out.println("Got ");

        assertTrue(avgMeanError < avgSumError, "Expected FMA mean to be more accurate on average");
    }
}
