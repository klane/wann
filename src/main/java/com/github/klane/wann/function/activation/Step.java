package com.github.klane.wann.function.activation;

public final class Step implements ActivationFunction {

    private final double min;
    private final double max;
    private final double threshold;

    public Step(final double min, final double max, final double threshold) {
        this.min = min;
        this.max = max;
        this.threshold = threshold;
    }

    @Override
    public double[] apply(final double[] input) {
        double[] output = new double[input.length];

        for (int i=0; i<input.length; i++) {
            output[i] = (input[i] >= this.threshold) ? this.max : this.min;
        }

        return output;
    }
}
