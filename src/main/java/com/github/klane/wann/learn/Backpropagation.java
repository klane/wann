package com.github.klane.wann.learn;

import com.github.klane.wann.core.Neuron;
import com.github.klane.wann.function.error.ErrorFunction;
import com.github.klane.wann.function.error.ErrorFunctions;
import weka.core.Instance;

import java.util.HashMap;
import java.util.Map;

public class Backpropagation extends LearningRule {

    public static final int DEFAULT_NUM_EPOCHS = 10;
    public static final double DEFAULT_LEARNING_RATE = 0.3;
    public static final ErrorFunction DEFAULT_ERROR_FUNCTION = ErrorFunctions.RMSE;

    private final double learningRate;
    private final Map<Neuron, Double> neuronError;

    public Backpropagation() {
        this(DEFAULT_NUM_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_ERROR_FUNCTION);
    }

    public Backpropagation(final int numEpochs) {
        this(numEpochs, DEFAULT_LEARNING_RATE, DEFAULT_ERROR_FUNCTION);
    }

    public Backpropagation(final int numEpochs, final double learningRate) {
        this(numEpochs, learningRate, DEFAULT_ERROR_FUNCTION);
    }

    public Backpropagation(final int numEpochs, final double learningRate, final ErrorFunction errorFunction) {
        super(numEpochs, errorFunction);
        this.learningRate = learningRate;
        this.neuronError = new HashMap<>();
    }

    @Override
    protected double[] learnInstance(final Instance instance) {
        final int size = super.output.size();
        final double[] error = new double[size];
        final double[] distribution = super.network.distributionForInstance(instance);

        for (int i=0; i<size; i++) {
            error[i] = (size > 1 ? (i == instance.classValue() ? 1 : 0) : instance.classValue()) - distribution[i];
        }

        this.updateOutputNeurons(error);
        this.updateHiddenNeurons();

        return error;
    }

    private void updateHiddenNeurons() {
        double error;

        for (int i=super.network.size()-2; i>0; i--) {
            for (Neuron neuron : super.network.getLayer(i)) {
                error = neuron.getOutputConnections().stream()
                        .mapToDouble(c -> this.neuronError.get(c.getToNeuron()) * c.getWeight()).sum();

                this.neuronError.put(neuron, error * neuron.getActivationFunction().derivative(neuron.getInput()));
                this.updateNeuronWeights(neuron);
            }
        }
    }

    private void updateOutputNeurons(final double[] error) {
        int i=0;

        for (Neuron neuron : super.output) {
            this.neuronError.put(neuron, error[i] * neuron.getActivationFunction().derivative(neuron.getInput()));
            this.updateNeuronWeights(neuron);
            i++;
        }
    }

    void updateNeuronWeights(final Neuron neuron) {
        final double error = this.neuronError.get(neuron);

        neuron.getInputConnections().forEach(c -> c.updateWeight(this.learningRate * error * c.getInput()));
    }
}
