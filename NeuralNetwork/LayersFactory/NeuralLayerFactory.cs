using NeuralNetwork.NeuralEventArgs;
using System.Collections.Generic;

namespace NeuralNetwork.LayersFactory
{
    internal class NeuralLayerFactory: ILayersFactory
    {
        private readonly uint _sensorsCount;
        private readonly uint[] _neurons;
        private readonly bool _isBiasNeurons;
        private readonly InitializerWeights _weightsInitializer;
        private readonly InitializerBias _biasInitializer;

        internal NeuralLayerFactory(uint sens, uint[] neurons, bool isBiasNeurons, InitializerWeights weightsInitializer, InitializerBias biasInitializer)
        {
            _sensorsCount = sens;
            _neurons = neurons;
            _isBiasNeurons = isBiasNeurons;
            _weightsInitializer = weightsInitializer;
            _biasInitializer = biasInitializer;
        }

        public LinkedList<ILayer> CreateLayers()
        {
            var _neuralLayers = new LinkedList<ILayer>();
            var _layersCount = _neurons.Length;
            var _synapsesCount = _sensorsCount;
            var _neuronsCount = _neurons[0];
            _neuralLayers.AddLast(new HiddenLayer(_synapsesCount, _neuronsCount, _isBiasNeurons, _weightsInitializer, _biasInitializer));

            for (int i = 1; i < _layersCount - 1; i++)
            {
                _synapsesCount = _neurons[i - 1];
                _neuronsCount = _neurons[i];
                _neuralLayers.AddLast(new HiddenLayer(_synapsesCount, _neuronsCount, _isBiasNeurons, _weightsInitializer, _biasInitializer));
            }

            _synapsesCount = _neurons[_layersCount - 2];
            _neuronsCount = _neurons[_layersCount - 1];
            _neuralLayers.AddLast(new OutputLayer(_synapsesCount, _neuronsCount, _isBiasNeurons, _weightsInitializer, _biasInitializer));

            for (var linkedListNode = _neuralLayers.First; linkedListNode.Next != null; linkedListNode = linkedListNode.Next)
            {
                var thisLayer = (NeuralLayer)linkedListNode.Value;
                var nextLayer = (NeuralLayer)linkedListNode.Next.Value;
                thisLayer.ActivatingLayer += (object sender, LayerEventArgs prevLayerEventArgs) =>
                {
                    nextLayer.Activate(sender, prevLayerEventArgs);
                };
                nextLayer.LearningLayer += (object sender, LayerEventArgs  nextLayerEventArgs) =>
                {
                    thisLayer.Learning(sender, nextLayerEventArgs);
                };
            }

            return _neuralLayers;
        }
    }
}
