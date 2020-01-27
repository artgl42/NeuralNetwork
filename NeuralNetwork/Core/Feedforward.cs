using System.Threading.Tasks;

namespace NeuralNetwork.Core
{
    static class Feedforward
    {
        internal static float[] GetOutputSignalOfSample(float[] inputSignal, float[][] neuronWeights, ActivationFunc activationFunc, float[] isBiasNeuron)
        {
            int _synapsesCount = neuronWeights.Length;
            int _neuronsCount = neuronWeights[0].Length;
            float[] _outputSignals = new float[_neuronsCount];
            float[][] _neuronWeightsT = TransposeArray(neuronWeights);

            for (int neuron = 0; neuron < _neuronsCount; neuron++)
            {
                float _inputSignalForNeuron = 0;
                for (int synapse = 0; synapse < _synapsesCount; synapse++)
                {
                    _inputSignalForNeuron += inputSignal[synapse] * _neuronWeightsT[neuron][synapse] + (isBiasNeuron != null ? isBiasNeuron[neuron] : 0);
                }
                _outputSignals[neuron] = NeuralMath.GetApproximateByFunction(_inputSignalForNeuron, activationFunc);
            }

            return _outputSignals;
        }

        internal static float[][] GetOutputSignalsOfAllSamples(float[][] inputSignals, float[][] neuronWeights, ActivationFunc activationFunc, float[] isBiasNeuron)
        {
            int _synapsesCount = neuronWeights.Length;
            int _neuronsCount = neuronWeights[0].Length;
            float[][] _outputSignals = new float[inputSignals.Length][];
            float[][] _neuronWeightsT = TransposeArray(neuronWeights);

            Parallel.For(0, inputSignals.Length, numberOfActiveDataset =>
            {
                float _inputSignalForNeuron;
                if (_outputSignals[numberOfActiveDataset] == null) _outputSignals[numberOfActiveDataset] = new float[_neuronsCount];

                for (int neuron = 0; neuron < _neuronsCount; neuron++)
                {
                    _inputSignalForNeuron = SumAllSynapseSignals(numberOfActiveDataset, neuron, _synapsesCount);
                    _outputSignals[numberOfActiveDataset][neuron] = NeuralMath.GetApproximateByFunction(_inputSignalForNeuron, activationFunc);
                }
            });

            return _outputSignals;

            float SumAllSynapseSignals(int currentSet, int neuron, int synapsesOfNeuron)
            {
                float _inputSignal = 0;
                for (int synapse = 0; synapse < synapsesOfNeuron; synapse++)
                {
                    _inputSignal += inputSignals[currentSet][synapse] * _neuronWeightsT[neuron][synapse] + (isBiasNeuron != null ? isBiasNeuron[neuron] : 0);
                }
                return _inputSignal;
            }
        }

        private static float[][] TransposeArray(float[][] arrayForTranspose)
        {
            float[][] _arrayT = new float[arrayForTranspose[0].Length][];
            for (int i = 0; i < arrayForTranspose.Length; i++)
            {
                for (int j = 0; j < arrayForTranspose[i].Length; j++)
                {
                    if (_arrayT[j] == null) _arrayT[j] = new float[arrayForTranspose.Length];
                    _arrayT[j][i] = arrayForTranspose[i][j];
                }
            }
            return _arrayT;
        }
    }
}
