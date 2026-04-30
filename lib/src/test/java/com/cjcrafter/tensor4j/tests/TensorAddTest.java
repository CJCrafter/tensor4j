package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TensorAddTest {

    private static final float EPS = 1e-6f;

    @Test
    void addSameShape() {
        Tensor a = TensorBuilder.builder()
                .shape(2, 2)
                .fromArray(1, 2, 3, 4);
        Tensor b = TensorBuilder.builder()
                .shape(2, 2)
                .fromArray(10, 20, 30, 40);
        Tensor c = a.add(b);

        assertEquals(11f, c.get(0, 0), EPS);
        assertEquals(22f, c.get(0, 1), EPS);
        assertEquals(33f, c.get(1, 0), EPS);
        assertEquals(44f, c.get(1, 1), EPS);
    }

    @Test
    void addSameShapeRowVectors() {
        Tensor a = TensorBuilder.builder()
                .shape(1, 3)
                .fromArray(1, 2, 3);
        Tensor b = TensorBuilder.builder()
                .shape(1, 3)
                .fromArray(4, 5, 6);
        Tensor c = a.add(b);

        assertEquals(5f, c.get(0, 0), EPS);
        assertEquals(7f, c.get(0, 1), EPS);
        assertEquals(9f, c.get(0, 2), EPS);
    }

    @Test
    void addBroadcastRowVector() {
        // [1, 2, 3]  +  [1, 2, 3]  =  [2,  4,  6]
        // [4, 5, 6]     [1, 2, 3]     [5,  7,  9]
        Tensor a = TensorBuilder.builder()
                .shape(2, 3)
                .fromArray(1, 2, 3, 4, 5, 6);
        Tensor b = TensorBuilder.builder()
                .shape(1, 3)
                .fromArray(1, 2, 3);
        Tensor c = a.add(b);

        assertEquals(2f, c.get(0, 0), EPS);
        assertEquals(4f, c.get(0, 1), EPS);
        assertEquals(6f, c.get(0, 2), EPS);
        assertEquals(5f, c.get(1, 0), EPS);
        assertEquals(7f, c.get(1, 1), EPS);
        assertEquals(9f, c.get(1, 2), EPS);
    }

    @Test
    void addBroadcastColVector() {
        // [1, 2, 3]  +  [10]  =  [11, 12, 13]
        // [4, 5, 6]     [20]     [24, 25, 26]
        Tensor a = TensorBuilder.builder()
                .shape(2, 3)
                .fromArray(1, 2, 3, 4, 5, 6);
        Tensor b = TensorBuilder.builder()
                .shape(2, 1)
                .fromArray(10, 20);
        Tensor c = a.add(b);

        assertEquals(11f, c.get(0, 0), EPS);
        assertEquals(12f, c.get(0, 1), EPS);
        assertEquals(13f, c.get(0, 2), EPS);
        assertEquals(24f, c.get(1, 0), EPS);
        assertEquals(25f, c.get(1, 1), EPS);
        assertEquals(26f, c.get(1, 2), EPS);
    }

    @Test
    void addBroadcastScalar() {
        Tensor a = TensorBuilder.builder()
                .shape(2, 2)
                .fromArray(1, 2, 3, 4);
        Tensor b = TensorBuilder.builder()
                .shape(1, 1)
                .fromArray(100);
        Tensor c = a.add(b);

        assertEquals(101f, c.get(0, 0), EPS);
        assertEquals(102f, c.get(0, 1), EPS);
        assertEquals(103f, c.get(1, 0), EPS);
        assertEquals(104f, c.get(1, 1), EPS);
    }

    @Test
    void addIncompatibleThrows() {
        Tensor a = TensorBuilder.builder()
                .shape(1, 3)
                .fromArray(1, 2, 3);
        Tensor b = TensorBuilder.builder()
                .shape(1, 2)
                .fromArray(1, 2);
        assertThrows(IllegalArgumentException.class, () -> a.add(b));
    }

    @Test
    void addResultShape() {
        Tensor a = TensorBuilder.builder().shape(2, 3).fromArray(1, 2, 3, 4, 5, 6);
        Tensor b = TensorBuilder.builder().shape(1, 3).fromArray(1, 2, 3);
        Tensor c = a.add(b);

        assertEquals(2, c.getShape().dimensions());
        assertEquals(2, c.getShape().dim(0));
        assertEquals(3, c.getShape().dim(1));
    }
}
