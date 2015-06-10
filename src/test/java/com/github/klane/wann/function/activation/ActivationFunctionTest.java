package com.github.klane.wann.function.activation;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public final class ActivationFunctionTest {

    private static final double TOLERANCE = 1E-4;

    @Test
    public void linear() {
        assertEquals(ActivationFunctions.LINEAR.applyAsDouble(-1), -1, 0);
        assertEquals(ActivationFunctions.LINEAR.applyAsDouble(0), 0, 0);
        assertEquals(ActivationFunctions.LINEAR.applyAsDouble(0.5), 0.5, 0);
        assertEquals(ActivationFunctions.LINEAR.applyAsDouble(1), 1, 0);
        assertEquals(ActivationFunctions.LINEAR.applyAsDouble(2), 2, 0);

        assertEquals(ActivationFunctions.LINEAR.derivative(-1), 1, 0);
        assertEquals(ActivationFunctions.LINEAR.derivative(0), 1, 0);
        assertEquals(ActivationFunctions.LINEAR.derivative(0.5), 1, 0);
        assertEquals(ActivationFunctions.LINEAR.derivative(1), 1, 0);
        assertEquals(ActivationFunctions.LINEAR.derivative(2), 1, 0);

        Linear linear = new Linear(2);

        assertEquals(linear.applyAsDouble(-1), -2, 0);
        assertEquals(linear.applyAsDouble(0), 0, 0);
        assertEquals(linear.applyAsDouble(0.5), 1, 0);
        assertEquals(linear.applyAsDouble(1), 2, 0);
        assertEquals(linear.applyAsDouble(2), 4, 0);

        assertEquals(linear.derivative(-1), 2, 0);
        assertEquals(linear.derivative(0), 2, 0);
        assertEquals(linear.derivative(0.5), 2, 0);
        assertEquals(linear.derivative(1), 2, 0);
        assertEquals(linear.derivative(2), 2, 0);
    }

    @Test
    public void logistic() {
        assertEquals(ActivationFunctions.SIGMOID.applyAsDouble(-10), 0, TOLERANCE);
        assertEquals(ActivationFunctions.SIGMOID.applyAsDouble(0), 0.5, 0);
        assertEquals(ActivationFunctions.SIGMOID.applyAsDouble(10), 1, TOLERANCE);

        assertEquals(ActivationFunctions.SIGMOID.derivative(-10), 0, TOLERANCE);
        assertEquals(ActivationFunctions.SIGMOID.derivative(0), 0.25, 0);
        assertEquals(ActivationFunctions.SIGMOID.derivative(10), 0, TOLERANCE);

        Logistic logistic = new Logistic(2, 1, 0.5);

        assertEquals(logistic.applyAsDouble(-10), 0, TOLERANCE);
        assertEquals(logistic.applyAsDouble(0.5), 1, 0);
        assertEquals(logistic.applyAsDouble(20), 2, TOLERANCE);
    }

    @Test
    public void ramp() {
        assertEquals(ActivationFunctions.RAMP.applyAsDouble(-1), 0, 0);
        assertEquals(ActivationFunctions.RAMP.applyAsDouble(0), 0, 0);
        assertEquals(ActivationFunctions.RAMP.applyAsDouble(0.5), 0.5, 0);
        assertEquals(ActivationFunctions.RAMP.applyAsDouble(1), 1, 0);
        assertEquals(ActivationFunctions.RAMP.applyAsDouble(2), 1, 0);

        assertEquals(ActivationFunctions.RAMP.derivative(-1), 1, 0);
        assertEquals(ActivationFunctions.RAMP.derivative(0), 1, 0);
        assertEquals(ActivationFunctions.RAMP.derivative(0.5), 1, 0);
        assertEquals(ActivationFunctions.RAMP.derivative(1), 1, 0);
        assertEquals(ActivationFunctions.RAMP.derivative(2), 1, 0);

        Ramp ramp = new Ramp(-1, 1.5, -2, 8);

        assertEquals(ramp.applyAsDouble(-1), -2, 0);
        assertEquals(ramp.applyAsDouble(0), 2, 0);
        assertEquals(ramp.applyAsDouble(0.5), 4, 0);
        assertEquals(ramp.applyAsDouble(1), 6, 0);
        assertEquals(ramp.applyAsDouble(2), 8, 0);

        assertEquals(ramp.derivative(-1), 1, 0);
        assertEquals(ramp.derivative(0), 1, 0);
        assertEquals(ramp.derivative(0.5), 1, 0);
        assertEquals(ramp.derivative(1), 1, 0);
        assertEquals(ramp.derivative(2), 1, 0);
    }

    @Test
    public void step() {
        assertEquals(ActivationFunctions.STEP.applyAsDouble(-1), 0, 0);
        assertEquals(ActivationFunctions.STEP.applyAsDouble(0), 1, 0);
        assertEquals(ActivationFunctions.STEP.applyAsDouble(0.5), 1, 0);
        assertEquals(ActivationFunctions.STEP.applyAsDouble(1), 1, 0);
        assertEquals(ActivationFunctions.STEP.applyAsDouble(2), 1, 0);

        assertEquals(ActivationFunctions.STEP.derivative(-1), 1, 0);
        assertEquals(ActivationFunctions.STEP.derivative(0), 1, 0);
        assertEquals(ActivationFunctions.STEP.derivative(0.5), 1, 0);
        assertEquals(ActivationFunctions.STEP.derivative(1), 1, 0);
        assertEquals(ActivationFunctions.STEP.derivative(2), 1, 0);

        Step step = new Step(-1, 2, 1);

        assertEquals(step.applyAsDouble(-1), -1, 0);
        assertEquals(step.applyAsDouble(0), -1, 0);
        assertEquals(step.applyAsDouble(0.5), -1, 0);
        assertEquals(step.applyAsDouble(1), 2, 0);
        assertEquals(step.applyAsDouble(2), 2, 0);

        assertEquals(step.derivative(-1), 1, 0);
        assertEquals(step.derivative(0), 1, 0);
        assertEquals(step.derivative(0.5), 1, 0);
        assertEquals(step.derivative(1), 1, 0);
        assertEquals(step.derivative(2), 1, 0);
    }

    @Test
    public void tanh() {
        assertEquals(ActivationFunctions.TANH.applyAsDouble(-10), -1, TOLERANCE);
        assertEquals(ActivationFunctions.TANH.applyAsDouble(0),0, 0);
        assertEquals(ActivationFunctions.TANH.applyAsDouble(10), 1, TOLERANCE);

        assertEquals(ActivationFunctions.TANH.derivative(-10), 0, TOLERANCE);
        assertEquals(ActivationFunctions.TANH.derivative(0), 1, 0);
        assertEquals(ActivationFunctions.TANH.derivative(10), 0, TOLERANCE);
    }
}
