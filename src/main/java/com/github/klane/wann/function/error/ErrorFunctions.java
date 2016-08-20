package com.github.klane.wann.function.error;

import lombok.AllArgsConstructor;

import java.util.stream.DoubleStream;

@AllArgsConstructor
public enum ErrorFunctions implements ErrorFunction {

    MSE(error -> DoubleStream.of(error).map(d -> d * d).average().getAsDouble()),
    RMSE(error -> Math.sqrt(DoubleStream.of(error).map(d -> d * d).average().getAsDouble()));

    private final ErrorFunction function;

    public double applyAsDouble(final double[] value) {
        return this.function.applyAsDouble(value);
    }
}
