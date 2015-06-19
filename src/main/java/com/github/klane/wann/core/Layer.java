package com.github.klane.wann.core;

import com.github.klane.wann.function.activation.ActivationFunction;
import com.github.klane.wann.function.input.InputFunction;
import com.google.common.base.Preconditions;
import javafx.util.Builder;

import java.util.*;
import java.util.stream.Collectors;

public final class Layer implements Iterable<Neuron> {

    private final String name;
    private final InputFunction inputFunction;
    private final ActivationFunction activationFunction;
    private final Layer previous;
    private Layer next;
    final Network network;
    final List<Neuron> neurons;

    private Layer(final LayerBuilder builder) {
        this.name = builder.name;
        this.network = builder.network;
        this.activationFunction = builder.activationFunction;

        if (this.network.size() > 0) {
            this.previous = this.network.getLayer(this.network.size() - 1);
            this.previous.next = this;
            this.inputFunction = builder.inputFunction;
        } else {
            this.previous = null;
            this.inputFunction = null;
        }

        this.neurons = builder.neurons.stream()
                .map(b -> b
                        .layer(this)
                        .name("L" + this.name.charAt(this.name.length() - 1) + "N" + (builder.neurons.indexOf(b) + 1))
                        .build())
                .collect(Collectors.toList());
    }

    public static LayerBuilder builder() {
        return new LayerBuilder();
    }

    public Neuron get(int i) {
        Preconditions.checkElementIndex(i, this.neurons.size(), "Invalid neuron index");
        return this.neurons.get(i);
    }

    public ActivationFunction getActivationFunction() {
        return this.activationFunction;
    }

    public String getName() {
        return this.name;
    }

    public Layer getNext() {
        return this.next;
    }

    public Layer getPrevious() {
        return this.previous;
    }

    @Override
    public Iterator<Neuron> iterator() {
        return this.neurons.iterator();
    }

    public int size() {
        return this.neurons.size();
    }

    @Override
    public String toString() {
        return this.name;
    }

    void calculate() {
        if (this.inputFunction != null) {
            this.neurons.forEach(n -> n.setInput(this.inputFunction.applyAsDouble(n.getInputConnections())));
        }

        this.neurons.forEach(n -> n.setOutput(this.activationFunction.applyAsDouble(n.getInput())));
    }

    public static final class LayerBuilder implements Builder<Layer> {

        private String name;
        private InputFunction inputFunction;
        private ActivationFunction activationFunction;
        private Network network;
        final List<Neuron.NeuronBuilder> neurons;

        private LayerBuilder() {
            this.neurons = new ArrayList<>();
        }

        public LayerBuilder activationFunction(final ActivationFunction activationFunction) {
            if (this.activationFunction == null) {
                this.activationFunction = activationFunction;
            }

            return this;
        }

        @Override
        public Layer build() {
            Preconditions.checkArgument(this.neurons.size() > 0, "Empty layer");
            Preconditions.checkNotNull(this.activationFunction, "Must specify a layer activation function");
            return new Layer(this);
        }

        public LayerBuilder inputFunction(final InputFunction inputFunction) {
            if (this.inputFunction == null) {
                this.inputFunction = inputFunction;
            }

            return this;
        }

        public LayerBuilder name(final String name) {
            this.name = name;
            return this;
        }

        public LayerBuilder neuron(final Collection<Neuron.NeuronBuilder> neurons) {
            this.neurons.addAll(neurons);
            return this;
        }

        public LayerBuilder neuron(final double... weights) {
            Neuron.NeuronBuilder builder = Neuron.builder();

            for (int i=0; i<weights.length; i++) {
                builder.connection(i, weights[i]);
            }

            return this.neuron(builder);
        }

        public LayerBuilder neuron(final Neuron.NeuronBuilder... neurons) {
            return this.neuron(Arrays.asList(neurons));
        }

        public LayerBuilder neuronWithBias(final double bias, final double... weights) {
            this.neuron(weights);
            this.neurons.get(this.neurons.size()-1).bias(bias);
            return this;
        }

        LayerBuilder network(final Network network) {
            this.network = network;
            return this;
        }
    }
}
