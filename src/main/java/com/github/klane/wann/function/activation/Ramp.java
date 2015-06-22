package com.github.klane.wann.function.activation;

public final class Ramp implements ActivationFunction {

    private final double xMin;
    private final double xMax;
    private final double yMin;
    private final double yMax;
    private final double slope;
    private final double intercept;

    public Ramp(final double xMin, final double xMax, final double yMin, final double yMax) {
        this.xMin = xMin;
        this.xMax = xMax;
        this.yMin = yMin;
        this.yMax = yMax;
        this.slope = (yMax - yMin) / (xMax - xMin);
        this.intercept = yMax - this.slope * xMax;
    }

    @Override
    public double[] apply(final double[] input) {
        double[] output = new double[input.length];

        for (int i=0; i<input.length; i++) {
            if (input[i] <= this.xMin) {
                output[i] = this.yMin;
            } else if (input[i] >= this.xMax) {
                output[i] = this.yMax;
            } else {
                output[i] = this.slope * input[i] + this.intercept;
            }
        }

        return output;
    }
}
