package com.github.klane.wekanet.function.activation;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;

public final class ActivationFunctionTest {

    private static final double[] INPUT_1 = new double[] {-1, 0, 0.5, 1, 2};
    private static final double[] INPUT_2 = new double[] {-10, 0, 10};
    private static final double[] ONES = new double[] {1, 1, 1, 1, 1};
    private static final double TOLERANCE = 1E-4;
    private ActivationFunction function;

    @Test
    public void linear() {
        function = ActivationFunctions.IDENTITY;

        assertArrayEquals(function.apply(INPUT_1), INPUT_1, 0);
        assertArrayEquals(function.derivative(INPUT_1), ONES, 0);

        function = new Linear(2);

        assertArrayEquals(function.apply(INPUT_1), new double[] {-2, 0, 1, 2, 4}, 0);
        assertArrayEquals(function.derivative(INPUT_1), new double[] {2, 2, 2, 2, 2}, 0);
    }

    @Test
    public void logistic() {
        function = ActivationFunctions.SIGMOID;

        assertArrayEquals(function.apply(INPUT_2), new double[] {0, 0.5, 1}, TOLERANCE);
        assertArrayEquals(function.derivative(INPUT_2), new double[] {0, 0.25, 0}, TOLERANCE);

        function = new Logistic(2, 1, 0.5);

        assertArrayEquals(function.apply(new double[] {-10, 0.5, 20}), new double[] {0, 1, 2}, TOLERANCE);
    }

    @Test
    public void ramp() {
        function = ActivationFunctions.RAMP;

        assertArrayEquals(function.apply(INPUT_1), new double[] {0, 0, 0.5, 1, 1}, 0);
        assertArrayEquals(function.derivative(INPUT_1), ONES, 0);

        function = new Ramp(-1, 1.5, -2, 8);

        assertArrayEquals(function.apply(INPUT_1), new double[] {-2, 2, 4, 6, 8}, 0);
        assertArrayEquals(function.derivative(INPUT_1), ONES, 0);
    }

    @Test
    public void softmax() {
        int n = 10;
        double[] output = new double[n];

        for (int i=0; i<n; i++) {
            output[i] = 1.0 / n;
        }

        function = ActivationFunctions.SOFTMAX;

        assertArrayEquals(function.apply(new double[n]), output, 0);
        assertArrayEquals(function.derivative(INPUT_1), ONES, 0);
    }

    @Test
    public void step() {
        function = ActivationFunctions.STEP;

        assertArrayEquals(function.apply(INPUT_1), new double[] {0, 1, 1, 1, 1}, 0);
        assertArrayEquals(function.derivative(INPUT_1), ONES, 0);

        function = new Step(-1, 2, 1);

        assertArrayEquals(function.apply(INPUT_1), new double[] {-1, -1, -1, 2, 2}, 0);
        assertArrayEquals(function.derivative(INPUT_1), ONES, 0);
    }

    @Test
    public void tanh() {
        function = ActivationFunctions.TANH;

        assertArrayEquals(function.apply(INPUT_2), new double[] {-1, 0, 1}, TOLERANCE);
        assertArrayEquals(function.derivative(INPUT_2), new double[] {0, 1, 0}, TOLERANCE);
    }
}
