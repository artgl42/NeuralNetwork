using NeuralNetwork.Core;
using NeuralNetwork.NeuralEventArgs;

namespace NeuralNetwork.LayersFactory
{
    internal class HiddenLayer : NeuralLayer
    {
        internal HiddenLayer(uint synapsesCount, uint neuronsCount, bool isBiasNeuron, InitializerWeights weightsInitializer, InitializerBias biasInitializer) 
            : base(synapsesCount, neuronsCount, isBiasNeuron, weightsInitializer, biasInitializer) { }

        protected override void CalculateDeltasErrorsForThisLayer(LayerEventArgs nextLayerEventArgs)
        {
            _DeltasErrorsOfNeurons = Backpropagation.GetDeltasHiddenLayer(
                _OutputSignals[(int)nextLayerEventArgs.NumberOfActiveDataset], 
                nextLayerEventArgs.NeuronsWeightsNextLayer, 
                nextLayerEventArgs.DeltasErrorsNextLayer, 
                nextLayerEventArgs.ActivationFunc);
        }
    }
}
