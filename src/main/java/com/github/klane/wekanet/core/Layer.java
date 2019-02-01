package com.github.klane.wekanet.core;

import com.github.klane.wekanet.function.activation.ActivationFunction;
import com.github.klane.wekanet.function.input.InputFunction;
import com.github.klane.wekanet.util.Builder;
import com.google.common.base.Preconditions;
import lombok.Getter;

import java.util.*;
import java.util.stream.Collectors;

public final class Layer implements Iterable<Neuron> {

    @Getter private final String name;
    @Getter private double[] input;
    @Getter private double[] output;
    private final InputFunction inputFunction;
    @Getter private final ActivationFunction activationFunction;
    @Getter private final Layer previous;
    @Getter private Layer next;
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

        this.input = new double[this.neurons.size()];
        this.output = new double[this.neurons.size()];
    }

    public static LayerBuilder builder() {
        return new LayerBuilder();
    }

    public Neuron get(int i) {
        Preconditions.checkElementIndex(i, this.neurons.size(), "Invalid neuron index");
        return this.neurons.get(i);
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
        int i;

        if (this.inputFunction != null) {
            i=0;

            for (Neuron n : this) {
                this.input[i++] = this.inputFunction.applyAsDouble(n.getInputConnections());
            }
        }

        i=0;
        this.output = this.activationFunction.apply(this.input);

        for (Neuron n : this) {
            n.setValue(this.output[i++]);
        }
    }

    void setInput(final double[] input) {
        this.input = Arrays.copyOf(input, input.length);
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
