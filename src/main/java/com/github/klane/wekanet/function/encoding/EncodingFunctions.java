package com.github.klane.wekanet.function.encoding;

import com.google.common.base.Preconditions;
import weka.core.Attribute;
import weka.core.Instance;

public enum EncodingFunctions implements EncodingFunction {

    BINARY {
        public double[] apply(Attribute attribute, Instance instance) {
            Preconditions.checkArgument(attribute.numValues() == 2, "Attribute must be binary");

            return new double[] {instance.value(attribute)};
        }
    },

    BINARY_1 {
        public double[] apply(Attribute attribute, Instance instance) {
            Preconditions.checkArgument(attribute.numValues() == 2, "Attribute must be binary");

            return new double[] {2*instance.value(attribute)-1};
        }
    },

    ONE_OF_C {
        public double[] apply(Attribute attribute, Instance instance) {
            Preconditions.checkArgument(attribute.isNominal(), "Attribute must be nominal");

            double[] code = new double[attribute.numValues()];
            code[(int) instance.value(attribute)] = 1;

            return code;
        }
    },

    ONE_OF_C_1 {
        public double[] apply(Attribute attribute, Instance instance) {
            Preconditions.checkArgument(attribute.isNominal(), "Attribute must be nominal");

            int index = (int) instance.value(attribute);
            double[] code = new double[attribute.numValues()-1];

            if (index < code.length) {
                code[index] = 1;
            }

            return code;
        }
    },

    ONE_OF_C_1_EFFECTS {
        public double[] apply(Attribute attribute, Instance instance) {
            Preconditions.checkArgument(attribute.isNominal(), "Attribute must be nominal");

            int index = (int) instance.value(attribute);
            double[] code = new double[attribute.numValues()-1];

            if (index == code.length) {
                for (int i=0; i<code.length; i++) {
                    code[i] = -1;
                }
            } else {
                code[index] = 1;
            }

            return code;
        }
    }
}
