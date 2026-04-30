package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TensorTest {

    @Test
    void testManualSeed() {
        Tensor4j.manualSeed(7L);
        Tensor a = Tensor4j.builder()
                .shape(128, 128)
                .randn();

        Tensor4j.manualSeed(7L);
        Tensor b = Tensor4j.builder()
                .shape(128, 128)
                .randn();

        Assertions.assertArrayEquals(a.getData(), b.getData());
    }

    @Test
    void testToString() {
        Tensor a = Tensor4j.builder()
                .shape(2, 4, 4)
                .rand();

        System.out.println(a);
    }
}
