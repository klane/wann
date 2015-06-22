package com.github.klane.wann.function.activation;

public final class Linear implements ActivationFunction {

    private final double slope;

    public Linear(final double slope) {
        this.slope = slope;
    }

    @Override
    public double[] apply(final double[] input) {
        double[] output = new double[input.length];

        for (int i=0; i<input.length; i++) {
            output[i] = this.slope * input[i];
        }

        return output;
    }

    @Override
    public double[] derivative(final double[] input) {
        double[] output = new double[input.length];

        for (int i=0; i<input.length; i++) {
            output[i] = this.slope;
        }

        return output;
    }
}
