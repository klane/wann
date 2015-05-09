package com.github.klane.wann.core;

import com.github.klane.wann.function.activation.ActivationFunction;
import com.github.klane.wann.function.input.InputFunction;
import com.google.common.base.Preconditions;

import java.util.*;
import java.util.stream.Collectors;

public final class Neuron implements Iterable<Connection> {

    private final String name;
    private final List<Connection> inputConnections;
    private final List<Connection> outputConnections;
    private final InputFunction inputFunction;
    private final ActivationFunction activationFunction;
    private double input;
    private double output;

    private Neuron(final Builder builder) {
        this.name = builder.name;
        this.inputConnections = new ArrayList<>();
        this.outputConnections = new ArrayList<>();

        if (!builder.connectionMap.isEmpty()) {
            this.inputConnections.addAll(builder.connectionMap.entrySet().stream()
                    .map(e -> new Connection(builder.layer.getPrevious().get(e.getKey()), this, e.getValue()))
                    .collect(Collectors.toList()));
        } else if (builder.layer != null && builder.layer.getPrevious() != null) {
            this.inputConnections.addAll(builder.layer.getPrevious().neurons.stream()
                    .map(n -> new Connection(n, this))
                    .collect(Collectors.toList()));
        }

        if (builder.biasFlag) {
            Double bias = builder.bias;

            if (bias == null) {
                this.inputConnections.add(new Connection(builder.layer.network.bias, this));
            } else {
                this.inputConnections.add(new Connection(builder.layer.network.bias, this, bias));
            }
        }

        this.inputConnections.forEach(c -> c.getFromNeuron().outputConnections.add(c));

        if (this.inputConnections.size() > 0) {
            Preconditions.checkNotNull(builder.inputFunction, "Must specify a neuron input function");
        }

        this.inputFunction = builder.inputFunction;
        this.activationFunction = builder.activationFunction;
    }

    public static Builder builder() {
        return new Builder();
    }

    public ActivationFunction getActivationFunction() {
        return this.activationFunction;
    }

    public double getInput() {
        return this.input;
    }

    public List<Connection> getInputConnections() {
        return new ArrayList<>(this.inputConnections);
    }

    public String getName() {
        return this.name;
    }

    public double getOutput() {
        return this.output;
    }

    public List<Connection> getOutputConnections() {
        return new ArrayList<>(this.outputConnections);
    }

    public double[] getWeights() {
        return this.inputConnections.stream().mapToDouble(Connection::getWeight).toArray();
    }

    @Override
    public Iterator<Connection> iterator() {
        return this.inputConnections.iterator();
    }

    @Override
    public String toString() {
        return this.name;
    }

    void calculate() {
        if (this.inputConnections.size() > 0) {
            this.input = this.inputFunction.applyAsDouble(this.inputConnections);
        }

        this.output = this.activationFunction.applyAsDouble(this.input);
    }

    void setInput(final double input) {
        this.input = input;
    }

    public static final class Builder extends WANNBuilder<Neuron, Builder> {

        private Layer layer;
        private final Map<Integer, Double> connectionMap;

        private Builder() {
            this.connectionMap = new LinkedHashMap<>();
        }

        @Override
        public Neuron build() {
            Preconditions.checkNotNull(this.activationFunction, "Must specify a neuron activation function");
            return new Neuron(this);
        }

        public Builder connection(final int fromNeuronIndex, final double weight) {
            this.connectionMap.put(fromNeuronIndex, weight);
            return this;
        }

        @Override
        Builder get() {
            return this;
        }

        Builder layer(final Layer layer) {
            Preconditions.checkNotNull(layer);
            this.layer = layer;
            return this;
        }
    }
}
