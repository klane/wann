package com.github.klane.wann.core;

import com.github.klane.wann.function.activation.ActivationFunction;
import com.github.klane.wann.function.input.InputFunction;
import com.google.common.base.Preconditions;
import javafx.util.Builder;

public abstract class WANNBuilder<T, U extends WANNBuilder<T, U>> implements Builder<T> {

    ActivationFunction activationFunction;
    Double bias;
    boolean biasFlag;
    InputFunction inputFunction;
    String name;

    public U activationFunction(final ActivationFunction activationFunction) {
        if (this.activationFunction == null) {
            this.activationFunction = activationFunction;
        }

        return this.get();
    }

    public U bias(final boolean biasFlag) {
        this.biasFlag = biasFlag;
        return this.get();
    }

    public U bias(final Double bias) {
        if (this.bias == null) {
            this.bias = bias;
            this.biasFlag = true;
        }

        return this.get();
    }

    @Override
    public abstract T build();

    public U inputFunction(final InputFunction inputFunction) {
        if (this.inputFunction == null) {
            this.inputFunction = inputFunction;
        }

        return this.get();
    }

    public U name(final String name) {
        Preconditions.checkNotNull(name);

        if (this.name == null) {
            this.name = name;
        }

        return this.get();
    }

    abstract U get();
}
