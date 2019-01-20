package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.ComputableGraphBuilder;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.joor.Reflect;

import java.util.*;
import java.util.function.Function;

import static java.util.stream.Collectors.toMap;

public class KeanuCompiledGraphBuilder implements ComputableGraphBuilder<ComputableGraph> {

    private StringBuilder computeSourceBuilder;
    private StringBuilder instanceVariableBuilder;
    private StringBuilder constructorBuilder;
    private Map<VariableReference, String> lookup;
    private Map<VariableReference, Object> variableValues;
    private Map<VariableReference, Object> constantValues;
    private List<VariableReference> outputs;

    private final String className = "CompiledKeanuGraph" + this.hashCode();

    public KeanuCompiledGraphBuilder() {
        computeSourceBuilder = new StringBuilder();
        instanceVariableBuilder = new StringBuilder();
        constructorBuilder = new StringBuilder();
        lookup = new HashMap<>();
        variableValues = new HashMap<>();
        constantValues = new HashMap<>();
        outputs = new ArrayList<>();
    }

    private void startSource(StringBuilder sb) {

        sb.append("package io.improbable.keanu.backend.keanu;\n");
        sb.append("import io.improbable.keanu.backend.VariableReference;\n");
        sb.append("import java.util.Collection;\n");
        sb.append("import java.util.Collections;\n");
        sb.append("import java.util.HashMap;\n");
        sb.append("import java.util.Map;\n");
        sb.append("import io.improbable.keanu.tensor.dbl.DoubleTensor;\n");

        sb.append("public final class " + className + " implements java.util.function.Function<Map<String, ?>, Map<String, ?>> {\n");
    }

    private void endSource(StringBuilder sb) {
        sb.append("Map<String, Object>  results = new HashMap<>();\n");
        sb.append("\n");

        for (VariableReference out : outputs) {
            String outputVariableName = toSourceVariableName(out);
            sb.append("results.put(\"" + out.toStringReference() + "\", " + outputVariableName + ");\n");
        }

        sb.append("return results;\n");

        sb.append("}\n}\n");
    }

    @Override
    public void createConstant(Vertex visiting) {

        Object value = visiting.getValue();
        String type = getType(value);
        String lookupName = visiting.getReference().toStringReference();
        String name = toSourceVariableName(visiting.getReference());

        instanceVariableBuilder.append("private final " + type + " " + name + ";\n");
        constructorBuilder.append(name + " = " + "(" + type + ")" + "constants.get(\"" + lookupName + "\");\n");

        lookup.put(visiting.getReference(), name);
        constantValues.put(visiting.getReference(), value);

    }

    @Override
    public void createVariable(Vertex visiting) {

        Object value = visiting.getValue();
        String variableType = getType(value);
        String variableName = toSourceVariableName(visiting.getReference());

        declareInput(variableType, variableName, visiting.getReference().toStringReference());

        lookup.put(visiting.getReference(), variableName);
        variableValues.put(visiting.getReference(), value);
    }

    private void declareInput(String type, String name, String inputName) {
        computeSourceBuilder.append("final " + type + " " + name + " = " + "(" + type + ")" + "inputs.get(\"" + inputName + "\");\n");
    }

    @Override
    public void create(Vertex visiting) {

        if (isConstant(visiting)) {
            createConstant(visiting);
            return;
        }

        KeanuVertexToTensorOpMapper.OpMapper opMapperFor = KeanuVertexToTensorOpMapper.getOpMapperFor(visiting.getClass());

        String variableType = getType(visiting);
        String name = toSourceVariableName(visiting.getReference());
        computeSourceBuilder.append("final " + variableType + " " + name + " = " + opMapperFor.apply(visiting, lookup) + ";\n");

        lookup.put(visiting.getReference(), name);
    }

    private boolean isConstant(Vertex v) {
        return v instanceof ConstantDoubleVertex || v instanceof ConstantIntegerVertex || v instanceof ConstantBooleanVertex;
    }

    private String getType(Object v) {
        return "DoubleTensor";
    }

    private String toSourceVariableName(VariableReference variableReference) {
        return "v_" + variableReference.toStringReference();
    }

    public void registerOutput(Vertex output) {
        outputs.add(output.getReference());
    }

    @Override
    public Collection<VariableReference> getLatentVariables() {
        return variableValues.keySet();
    }

    @Override
    public VariableReference add(VariableReference left, VariableReference right) {
        return null;
    }

    @Override
    public void connect(Map<? extends Vertex<?>, ? extends Vertex<?>> connections) {
        connections.forEach((to, from) ->
            lookup.put(from.getReference(), lookup.get(to.getReference()))
        );
    }

    @Override
    public ComputableGraph build() {
        StringBuilder stringBuilder = new StringBuilder();

        startSource(stringBuilder);

        stringBuilder.append(instanceVariableBuilder);

        stringBuilder.append("public " + className + "(final Map<String, ?> constants) {\n");
        stringBuilder.append(constructorBuilder);
        stringBuilder.append("}\n");

        stringBuilder.append("public Map<String, ?> apply(Map<String, ?> inputs) {\n");
        stringBuilder.append(computeSourceBuilder);

        endSource(stringBuilder);

        String source = stringBuilder.toString();

//        System.out.println(source);

        return compile(source);
    }

    private ComputableGraph compile(String source) {

        Map<String, ?> constantsByString = constantValues.entrySet().stream()
            .collect(toMap(e -> e.getKey().toStringReference(), Map.Entry::getValue));

        Function<Map<String, ?>, Map<String, ?>> computeFunction = Reflect.compile(
            "io.improbable.keanu.backend.keanu." + className,
            source
        ).create(constantsByString).get();

        return new WrappedCompiledGraph(computeFunction, constantValues, outputs);
    }

    private static class WrappedCompiledGraph implements ComputableGraph {

        private Map<String, VariableReference> outputsByString;
        private Function<Map<String, ?>, Map<String, ?>> computeFunction;

        private WrappedCompiledGraph(Function<Map<String, ?>, Map<String, ?>> computeFunction,
                                     Map<VariableReference, Object> constantValues,
                                     List<VariableReference> outputs) {
            this.computeFunction = computeFunction;
            this.outputsByString = outputs.stream()
                .collect(toMap(VariableReference::toStringReference, output -> output));
        }

        @Override
        public Map<VariableReference, ?> compute(Map<VariableReference, ?> inputs, Collection<VariableReference> outputs) {

            final Map<String, Object> inputsByString = new HashMap<>();

            for (Map.Entry<VariableReference, ?> input : inputs.entrySet()) {
                inputsByString.put(input.getKey().toStringReference(), input.getValue());
            }

            final Map<String, ?> resultsByString = computeFunction.apply(inputsByString);

            final Map<VariableReference, Object> results = new HashMap<>();

            for (Map.Entry<String, ?> result : resultsByString.entrySet()) {
                results.put(outputsByString.get(result.getKey()), result.getValue());
            }

            return results;
        }

        @Override
        public <T> T getInput(VariableReference input) {
            return null;
        }
    }
}