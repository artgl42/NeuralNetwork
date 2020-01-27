using NeuralNetwork.Core;
using NeuralNetwork.NeuralEventArgs;

namespace NeuralNetwork.LayersFactory
{
    internal class OutputLayer : NeuralLayer
    {
        internal OutputLayer(uint synapsesCount, uint neuronsCount, bool isBiasNeuron, InitializerWeights weightsInitializer, InitializerBias biasInitializer) 
            : base(synapsesCount, neuronsCount, isBiasNeuron, weightsInitializer, biasInitializer) { }

        protected override void CalculateDeltasErrorsForThisLayer(LayerEventArgs nextLayerEventArgs)
        {
            _DeltasErrorsOfNeurons = Backpropagation.GetDeltasOutputLayer(
                _OutputSignals[(int)nextLayerEventArgs.NumberOfActiveDataset],
                nextLayerEventArgs.ExpectedSignalsOutLayer[(int)nextLayerEventArgs.NumberOfActiveDataset],
                nextLayerEventArgs.ActivationFunc);
        }
    }
}
