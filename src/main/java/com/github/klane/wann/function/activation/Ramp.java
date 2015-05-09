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
    public double applyAsDouble(final double input) {
        if (input <= this.xMin) {
            return this.yMin;
        } else if (input >= this.xMax) {
            return this.yMax;
        }

        return this.slope * input + this.intercept;
    }
}
