package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.Shape;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ShapeTest {

    @Test
    void rejects1D() {
        assertThrows(IllegalArgumentException.class, () -> new Shape(5));
    }

    @Test
    void rowVectorShape() {
        Shape s = new Shape(1, 5);
        assertEquals(2, s.dimensions());
        assertEquals(5, s.numel());
    }

    @Test
    void colVectorShape() {
        Shape s = new Shape(5, 1);
        assertEquals(2, s.dimensions());
        assertEquals(5, s.numel());
    }

    @Test
    void matrixShape() {
        Shape s = new Shape(3, 4);
        assertEquals(2, s.dimensions());
        assertEquals(12, s.numel());
    }

    @Test
    void threeDShape() {
        Shape s = new Shape(2, 3, 4);
        assertEquals(3, s.dimensions());
        assertEquals(24, s.numel());
    }

    @Test
    void contiguousStridesRowVector() {
        int[] strides = Shape.contiguousStridesFrom(new int[]{1, 5});
        assertArrayEquals(new int[]{5, 1}, strides);
    }

    @Test
    void contiguousStridesMatrix() {
        int[] strides = Shape.contiguousStridesFrom(new int[]{3, 4});
        assertArrayEquals(new int[]{4, 1}, strides);
    }

    @Test
    void contiguousStrides3D() {
        int[] strides = Shape.contiguousStridesFrom(new int[]{2, 3, 4});
        assertArrayEquals(new int[]{12, 4, 1}, strides);
    }
}
