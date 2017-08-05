package com.github.klane.wann.learn;

import static org.junit.Assert.assertEquals;

import com.github.klane.wann.core.Connection;
import com.github.klane.wann.core.Layer;
import com.github.klane.wann.core.Network;
import com.github.klane.wann.core.Neuron;
import com.github.klane.wann.function.activation.ActivationFunctions;
import com.github.klane.wann.util.DataSetBuilder;
import com.github.klane.wann.util.InstanceBuilder;
import com.github.klane.wann.util.WekaUtils;
import org.junit.Test;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.List;

public final class BackpropagationTest {

    private static final double TOLERANCE = 1E-4;

    @Test
    public void backpropagation() {
        Layer layer;
        double[] layerInput;
        Neuron neuron;
        List<Connection> connections;
        double x = 0.35;
        double y = 0.9;
        double z = 0.5;
        double[][] input = {{x, y}, {0.755, 0.68}, {0.8014}};
        double[][] output = {{x, y}, {0.6803, 0.6637}, {0.6903}};
        double[][][] weights = {{{0.09916, 0.7978}, {0.3972, 0.5928}}, {{0.272392, 0.87305}}};

        Instances instances = new DataSetBuilder()
                .name("Test")
                .addAttribute(new Attribute("X"))
                .addAttribute(new Attribute("Y"))
                .addAttribute(new Attribute("Z"))
                .addInstance(new InstanceBuilder().values(x, y, z))
                .build();

        Network network = Network.builder()
                .name(instances.relationName())
                .inputs(WekaUtils.attributesIgnoreClass(instances))
                .layer(Layer.builder().activationFunction(ActivationFunctions.SIGMOID)
                        .neuron(0.1, 0.8)
                        .neuron(0.4, 0.6))
                .layer(Layer.builder().activationFunction(ActivationFunctions.SIGMOID)
                        .neuron(0.3, 0.9))
                .learningRule(new Backpropagation(1, 1.0))
                .build();

        network.buildClassifier(instances);

        for (int i=0; i<network.size(); i++) {
            layer = network.getLayer(i);
            layerInput = layer.getInput();

            for (int j=0; j<layer.size(); j++) {
                neuron = layer.get(j);

                assertEquals(layerInput[j], input[i][j], TOLERANCE);
                assertEquals(neuron.getValue(), output[i][j], TOLERANCE);

                if (i > 0) {
                    connections = neuron.getInputConnections();

                    for (int k=0; k<connections.size(); k++) {
                        assertEquals(connections.get(k).getWeight(), weights[i-1][j][k], TOLERANCE);
                    }
                }
            }
        }
    }
}
