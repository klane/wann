package com.github.klane.wekanet.function.activation;

import lombok.AllArgsConstructor;

@AllArgsConstructor
public enum ActivationFunctions implements ActivationFunction {

    IDENTITY(new Linear(1)),
    RAMP(new Ramp(0, 1, 0, 1)),
    SIGMOID(new Logistic(1, 1, 0)),
    SOFTMAX(input -> {
        double sum = 0;
        double[] output = new double[input.length];

        for (int i=0; i<input.length; i++) {
            output[i] = Math.exp(input[i]);
            sum += output[i];
        }

        for (int i=0; i<input.length; i++) {
            output[i] /= sum;
        }

        return output;
    }),
    STEP(new Step(0, 1, 0)),
    TANH(new ActivationFunction() {
        @Override
        public double[] apply(final double[] input) {
            double[] output = new double[input.length];

            for (int i=0; i<input.length; i++) {
                output[i] = Math.tanh(input[i]);
            }

            return output;
        }

        @Override
        public double[] derivative(final double[] input) {
            double[] output = this.apply(input);

            for (int i=0; i<input.length; i++) {
                output[i] = 1.0 - output[i] * output[i];
            }

            return output;
        }
    });

    private final ActivationFunction function;

    public double[] apply(final double[] input) {
        return this.function.apply(input);
    }

    public double[] derivative(final double[] input) {
        return this.function.derivative(input);
    }
}
