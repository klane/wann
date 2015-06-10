package com.github.klane.wann.function.error;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import java.util.stream.DoubleStream;

public final class ErrorFunctionTest {

    private static final double[] X = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    private static final double SQUARE_SUM = DoubleStream.of(X).map(x -> x * x).sum();

    @Test
    public void MSE() {
        assertEquals(ErrorFunctions.MSE.applyAsDouble(X), SQUARE_SUM / X.length, 0);
    }

    @Test
    public void RMSE() {
        assertEquals(ErrorFunctions.RMSE.applyAsDouble(X), Math.sqrt(SQUARE_SUM / X.length), 0);
    }
}
