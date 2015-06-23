package com.github.klane.wann.function.input;

import static org.junit.Assert.assertEquals;

import com.github.klane.wann.core.Connection;
import com.github.klane.wann.core.Layer;
import com.github.klane.wann.core.Network;
import com.github.klane.wann.core.Neuron;
import com.github.klane.wann.function.activation.ActivationFunctions;
import org.bitbucket.klane.weka.DataSetBuilder;
import org.bitbucket.klane.weka.WekaUtils;
import org.junit.Before;
import org.junit.Test;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.util.List;

public final class InputFunctionTest {

    private static final File ARFF = new File("./src/test/resources/XOR_nominal.arff");
    private static final Instances DATASET = new DataSetBuilder().arffLoad(ARFF).build();
    private Network network;
    private List<Connection> connections;

    @Before
    public void setUp() {
        network = Network.builder()
                .name(DATASET.relationName())
                .inputs(WekaUtils.attributesIgnoreClass(DATASET))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuron(
                        Neuron.builder().connection(0, 1.0).bias(-1.0),
                        Neuron.builder().connection(0, 1.0).connection(1, 1.0).bias(-2.0),
                        Neuron.builder().connection(1, 1.0).bias(-1.0)))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuronWithBias(-1.0, 1.0, -2.0, 1.0))
                .build();

        connections = network.getOutputLayer().get(0).getInputConnections();
    }

    @Test
    public void weightedMax() {
        double max;

        for (Instance i : WekaUtils.toList(DATASET)) {
            network.distributionForInstance(i);
            max = 0;

            for (Connection c : connections) {
                max = Math.max(max, c.getWeightedValue());
            }

            assertEquals(InputFunctions.WEIGHTED_MAX.applyAsDouble(connections), max, 0);
        }
    }

    @Test
    public void weightedMean() {
        double sum;

        for (Instance i : WekaUtils.toList(DATASET)) {
            network.distributionForInstance(i);
            sum = 0;

            for (Connection c : connections) {
                sum += c.getWeightedValue();
            }

            assertEquals(InputFunctions.WEIGHTED_MEAN.applyAsDouble(connections), sum / connections.size(), 0);
        }
    }

    @Test
    public void weightedSum() {
        double sum;

        for (Instance i : WekaUtils.toList(DATASET)) {
            network.distributionForInstance(i);
            sum = 0;

            for (Connection c : connections) {
                sum += c.getWeightedValue();
            }

            assertEquals(InputFunctions.WEIGHTED_SUM.applyAsDouble(connections), sum, 0);
        }
    }
}
