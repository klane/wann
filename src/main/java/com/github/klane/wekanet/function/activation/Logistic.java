package com.github.klane.wekanet.function.activation;

import lombok.AllArgsConstructor;

@AllArgsConstructor
public final class Logistic implements ActivationFunction {

    private final double max;
    private final double slope;
    private final double midpoint;

    @Override
    public double[] apply(final double[] input) {
        double[] output = new double[input.length];

        for (int i=0; i<input.length; i++) {
            output[i] = this.max / (1.0 + Math.exp(-this.slope * (input[i] - this.midpoint)));
        }

        return output;
    }

    @Override
    public double[] derivative(final double[] input) {
        double[] output = this.apply(input);

        for (int i=0; i<input.length; i++) {
            output[i] = this.slope * output[i] * (1.0 - output[i] / this.max);
        }

        return output;
    }
}
