package com.github.klane.wann.function.activation;

import java.util.function.Function;

public interface ActivationFunction extends Function<double[], double[]> {

    default double[] derivative(final double[] input) {
        double[] output = new double[input.length];

        for (int i=0; i<input.length; i++) {
            output[i] = 1.0;
        }

        return output;
    }
}
