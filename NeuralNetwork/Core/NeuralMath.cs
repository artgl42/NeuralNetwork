using System;

namespace NeuralNetwork.Core
{
    static class NeuralMath
    {
        private static readonly Random _random = new Random();

        internal static float[][] InitializeNeuronWeights(uint synapsesCount, uint neuronsCount, InitializerWeights weightsInitializer)
        {
            float[][] _randomWeights = new float[synapsesCount][];

            for (int synapse = 0; synapse < synapsesCount; synapse++)
            {
                if (_randomWeights[synapse] == null) _randomWeights[synapse] = new float[neuronsCount];
                for (int neuron = 0; neuron < neuronsCount; neuron++)
                {
                    _randomWeights[synapse][neuron] = GetWeights(weightsInitializer);
                }
            }

            return _randomWeights;

            float GetWeights(InitializerWeights initializer)
            {
                float _weight = 0;
                switch (initializer)
                {
                    case InitializerWeights.Zeros:
                        _weight = 0F;
                        break;
                    case InitializerWeights.Ones:
                        _weight = 1F;
                        break;
                    case InitializerWeights.Random:
                        _weight = (float)Math.Round(_random.NextDouble(), 5);
                        break;
                    case InitializerWeights.XavierUniform:
                        var max = Math.Sqrt(6 / (synapsesCount + neuronsCount));
                        var min = -1 * max;
                        var range = 2 * max;
                        _weight = (float)Math.Round(min + _random.NextDouble() * range, 5);
                        break;
                }
                return _weight;
            }
        }

        internal static float[] InitializeBiasWeights(uint neuronsCount, InitializerBias biasInitializer)
        {
            float[] _randomWeights = new float[neuronsCount];

            for (int neuron = 0; neuron < neuronsCount; neuron++)
            {
                _randomWeights[neuron] = GetWeights(biasInitializer);
            }

            return _randomWeights;

            float GetWeights(InitializerBias initializer)
            {
                float _weight = 0;
                switch (initializer)
                {
                    case InitializerBias.Zeros:
                        _weight = 0F;
                        break;
                    case InitializerBias.Ones:
                        _weight = 1F;
                        break;
                    case InitializerBias.Random:
                        _weight = (float)Math.Round(_random.NextDouble(), 5);
                        break;
                }
                return _weight;
            }
        }

        internal static float GetApproximateByFunction(float variableOfApproximate, ActivationFunc activationFunc)
        {
            float _result = 0;
            switch (activationFunc)
            {
                case ActivationFunc.Identity:
                    _result = variableOfApproximate;
                    break;
                case ActivationFunc.Sigmoid:
                    _result = (float)(1 / (1 + Math.Exp(-variableOfApproximate)));
                    break;
                case ActivationFunc.TanH:
                    _result = (float)((Math.Exp(2 * variableOfApproximate) - 1) / (Math.Exp(2 * variableOfApproximate) + 1));
                    break;
                case ActivationFunc.ReLU:
                    _result = variableOfApproximate > 0 ? variableOfApproximate : 0;
                    break;
                case ActivationFunc.Gaussian:
                    _result = (float)Math.Exp(-Math.Pow(variableOfApproximate, 2));
                    break;
            }
            return _result;
        }

        internal static float GetDerivativeByFunction(float variableOfDerivative, ActivationFunc activationFunc)
        {
            float _result = 0;
            switch (activationFunc)
            {
                case ActivationFunc.Identity:
                    _result = 1;
                    break;
                case ActivationFunc.Sigmoid:
                    _result = (1 - variableOfDerivative) * variableOfDerivative;
                    break;
                case ActivationFunc.TanH:
                    _result = (float)(1 - Math.Pow(variableOfDerivative, 2));
                    break;
                case ActivationFunc.ReLU:
                    _result = variableOfDerivative > 0 ? 1 : 0;
                    break;
                case ActivationFunc.Gaussian:
                    _result = (float)(-2 * variableOfDerivative * Math.Exp(-Math.Pow(variableOfDerivative, 2)));
                    break;
            }
            return _result;
        }

        internal static float GetTotalLoss(float[][] outputSignals, float[][] expectedSignals, LossFunc lossFunc)
        {
            float _totalLoss = 0;

            switch (lossFunc)
            {
                case LossFunc.MSE:
                    for (int _sample = 0; _sample < outputSignals.Length; _sample++)
                    {
                        for (int _neuron = 0; _neuron < outputSignals[0].Length; _neuron++)
                        {
                            _totalLoss += (float)Math.Pow(expectedSignals[_sample][_neuron] - outputSignals[_sample][_neuron], 2);
                        }
                    }
                    _totalLoss = _totalLoss / outputSignals.Length;
                    break;

                case LossFunc.RootMSE:
                    _totalLoss = (float)Math.Sqrt(GetTotalLoss(outputSignals, expectedSignals, LossFunc.MSE));
                    break;

                case LossFunc.Arctan:
                    for (int _sample = 0; _sample < outputSignals.Length; _sample++)
                    {
                        for (int _neuron = 0; _neuron < outputSignals[0].Length; _neuron++)
                        {
                            _totalLoss += (float)Math.Pow(Math.Atan(expectedSignals[_sample][_neuron] - outputSignals[_sample][_neuron]), 2);
                        }
                    }
                    _totalLoss = _totalLoss / outputSignals.Length;
                    break;
            }

            return _totalLoss;
        }
    }
}
