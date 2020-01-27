using System.Threading.Tasks;

namespace NeuralNetwork.Core
{
    static class Backpropagation
    {
        internal static float[] GetDeltasOutputLayer(float[] outputSignals, float[] expectedSignals, ActivationFunc activationFunc)
        {
            int _neuronsCountThisLayer = outputSignals.Length;
            float[] _deltasErrors = new float[_neuronsCountThisLayer];
            float _errorOutSignal;
            float _derivativeActivFunc;

            for (int _neuron = 0; _neuron < _neuronsCountThisLayer; _neuron++)
            {
                _errorOutSignal = expectedSignals[_neuron] - outputSignals[_neuron];
                _derivativeActivFunc = NeuralMath.GetDerivativeByFunction(outputSignals[_neuron], activationFunc);
                _deltasErrors[_neuron] = _errorOutSignal * _derivativeActivFunc;
            }

            return _deltasErrors;
        }

        internal static float[] GetDeltasHiddenLayer(float[] outputSignals, float[][] weightsNextLayer, float[] deltasErrorsNextLayer, ActivationFunc activationFunc)
        {
            int _neuronsCountThisLayer = outputSignals.Length;
            int _neuronsCountNextLayer = deltasErrorsNextLayer.Length;
            float[] _deltasErrors = new float[outputSignals.Length];
            float _sumWeightsAndDeltasNextLayer;
            float _derivativeActivFunc;

            for (int _neuron = 0; _neuron < _neuronsCountThisLayer; _neuron++)
            {
                _sumWeightsAndDeltasNextLayer = GetSumWeightsAndDeltasNextLayer(_neuron);
                _derivativeActivFunc = NeuralMath.GetDerivativeByFunction(outputSignals[_neuron], activationFunc);
                _deltasErrors[_neuron] = _sumWeightsAndDeltasNextLayer * _derivativeActivFunc;
            }

            return _deltasErrors;

            float GetSumWeightsAndDeltasNextLayer(int neuronPrevLayer)
            {
                float _sumWeightsAndDeltas = 0;
                for (int _neuronNextLayer = 0; _neuronNextLayer < _neuronsCountNextLayer; _neuronNextLayer++)
                {
                    _sumWeightsAndDeltas += deltasErrorsNextLayer[_neuronNextLayer] * weightsNextLayer[neuronPrevLayer][_neuronNextLayer];
                }
                return _sumWeightsAndDeltas;
            }
        }

        internal static float[][] GetGradientsForWeights(float[] inputSignals, float[] deltasErrorThisLayer)
        {
            int _synapsesCount = inputSignals.Length;
            int _neuronsCount = deltasErrorThisLayer.Length;
            float[][] _gradientsForWeights = new float[_synapsesCount][];

            Parallel.For(0, _synapsesCount, _synapse =>
            {
                if (_gradientsForWeights[_synapse] == null) _gradientsForWeights[_synapse] = new float[_neuronsCount];
                float _inputSignal = inputSignals[_synapse];

                for (int _neuron = 0; _neuron < _neuronsCount; _neuron++)
                {
                    _gradientsForWeights[_synapse][_neuron] = _inputSignal * deltasErrorThisLayer[_neuron];
                }
            });

            return _gradientsForWeights;
        }

        internal static float[][] GetUpdatesForWeights(float[][] gradientsForWeights, float[][] prevUpdatesForWeights, float learningRate, float momentumRate, LearningOptimizing? learningOptimizing)
        {
            int _neuronsCountThisLayer = gradientsForWeights[0].Length;
            int _synapsesCountThisLayer = gradientsForWeights.Length;
            float[][] _updatesForWeights = new float[_synapsesCountThisLayer][];
            if (prevUpdatesForWeights == null)
            {
                prevUpdatesForWeights = new float[_synapsesCountThisLayer][];
                for (int synapse = 0; synapse < _synapsesCountThisLayer; synapse++)
                {
                    prevUpdatesForWeights[synapse] = new float[_neuronsCountThisLayer];
                }
            }

            Parallel.For(0, _synapsesCountThisLayer, _synapseOfNeuron =>
            {
                if (_updatesForWeights[_synapseOfNeuron] == null) _updatesForWeights[_synapseOfNeuron] = new float[_neuronsCountThisLayer];

                for (int _neuronThisLayer = 0; _neuronThisLayer < _neuronsCountThisLayer; _neuronThisLayer++)
                {
                    _updatesForWeights[_synapseOfNeuron][_neuronThisLayer] = CalcUpdateForWeight(_synapseOfNeuron, _neuronThisLayer);
                }
            });

            return _updatesForWeights;

            float CalcUpdateForWeight(int synapse, int neuron)
            {
                float _updateForWeight = 0;
                switch (learningOptimizing)
                {
                    case null:
                        _updateForWeight = gradientsForWeights[synapse][neuron] * learningRate;
                        break;
                    case LearningOptimizing.NAG:
                        _updateForWeight = gradientsForWeights[synapse][neuron] * learningRate + prevUpdatesForWeights[synapse][neuron] * momentumRate;
                        break;
                }
                return _updateForWeight;
            }
        }

        internal static float[] GetUpdatesForBias(float[] deltasErrorThisLayer, float[] prevUpdatesForBias, float learningRate, float momentumRate, LearningOptimizing? learningOptimizing)
        {
            int _neuronsCountThisLayer = deltasErrorThisLayer.Length;
            float[] _updatesForBias = new float[_neuronsCountThisLayer];
            if (prevUpdatesForBias == null) prevUpdatesForBias = new float[_neuronsCountThisLayer];

            for (int _neuronThisLayer = 0; _neuronThisLayer < _neuronsCountThisLayer; _neuronThisLayer++)
            {
                _updatesForBias[_neuronThisLayer] = CalcUpdateForWeight(_neuronThisLayer);
            }

            return _updatesForBias;

            float CalcUpdateForWeight(int neuron)
            {
                float _updateForWeight = 0;
                switch (learningOptimizing)
                {
                    case null:
                        _updateForWeight = deltasErrorThisLayer[neuron] * learningRate;
                        break;
                    case LearningOptimizing.NAG:
                        _updateForWeight = deltasErrorThisLayer[neuron] * learningRate + prevUpdatesForBias[neuron] * momentumRate;
                        break;
                }
                return _updateForWeight;
            }
        }
    }
}
