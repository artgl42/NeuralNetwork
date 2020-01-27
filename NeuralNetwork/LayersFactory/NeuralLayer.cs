using NeuralNetwork.Core;
using NeuralNetwork.NeuralEventArgs;
using System;

namespace NeuralNetwork.LayersFactory
{
    abstract class NeuralLayer : ILayer
    {
        protected uint _SynapsesCount { get; set; }
        protected uint _NeuronsCount { get; set; }
        protected float[][] _InputSignals { get; set; }
        protected float[][] _OutputSignals { get; set; }
        protected float[][] _NeuronWeights { get; set; }
        protected float[] _DeltasErrorsOfNeurons { get; set; }
        protected float[][] _GradientsForWeights { get; set; }
        protected float[][] _UpdatesForWeights { get; set; }
        protected float[] _BiasWeights { get; set; }
        protected float[] _UpdatesForBias { get; set; }

        public event EventHandler<LayerEventArgs> ActivatingLayer;
        public event EventHandler<LayerEventArgs> LearningLayer;

        internal NeuralLayer(uint synapsesCount, uint neuronsCount, bool isBiasNeuron, InitializerWeights weightsInitializer, InitializerBias biasInitializer)
        {
            _SynapsesCount = synapsesCount;
            _NeuronsCount = neuronsCount;          
            _NeuronWeights = NeuralMath.InitializeNeuronWeights(_SynapsesCount, _NeuronsCount, weightsInitializer);
            if (isBiasNeuron) _BiasWeights = NeuralMath.InitializeBiasWeights(neuronsCount, biasInitializer);
        }

        public virtual void Activate(object sender, LayerEventArgs prevLayerEventArgs)
        {
            if (prevLayerEventArgs.NumberOfActiveDataset == null)
            {
                ActionByFull(prevLayerEventArgs);   
            }
            else
            {
                ActionBySelect(prevLayerEventArgs);
            }
            OnActivatingNextLayer(prevLayerEventArgs);
        }

        public virtual void Learning(object sender, LayerEventArgs nextLayerEventArgs)
        {
            CalculateDeltasErrorsForThisLayer(nextLayerEventArgs);
            OnLearningPrevLayer(nextLayerEventArgs);
            UpdateWeightsForNeurons(nextLayerEventArgs);
            if (_BiasWeights != null) UpdateWeightsForBias(nextLayerEventArgs);  
        }

        protected virtual void ActionByFull(LayerEventArgs layerEventArgs)
        {
            _InputSignals = layerEventArgs.OutputsSignalsPrevLayer;
            _OutputSignals = Feedforward.GetOutputSignalsOfAllSamples(
                _InputSignals, 
                _NeuronWeights, 
                layerEventArgs.ActivationFunc, 
                _BiasWeights);
        }

        protected virtual void ActionBySelect(LayerEventArgs layerEventArgs)
        {
            var _numOfActiveDataset = (int)layerEventArgs.NumberOfActiveDataset;
            _InputSignals[_numOfActiveDataset] = layerEventArgs.OutputsSignalsPrevLayer[_numOfActiveDataset];
            _OutputSignals[_numOfActiveDataset] = Feedforward.GetOutputSignalOfSample(
                _InputSignals[_numOfActiveDataset], 
                _NeuronWeights, 
                layerEventArgs.ActivationFunc, 
                _BiasWeights);
        }

        protected virtual void OnActivatingNextLayer(LayerEventArgs layerEventArgs)
        {
            layerEventArgs.OutputsSignalsPrevLayer = _OutputSignals;
            ActivatingLayer?.Invoke(this, layerEventArgs);
        }

        protected virtual void OnLearningPrevLayer(LayerEventArgs layerEventArgs)
        {
            layerEventArgs.DeltasErrorsNextLayer = _DeltasErrorsOfNeurons;
            layerEventArgs.NeuronsWeightsNextLayer = _NeuronWeights;
            LearningLayer?.Invoke(this, layerEventArgs);
        }

        protected virtual void UpdateWeightsForNeurons(LayerEventArgs layerEventArgs)
        {
            _GradientsForWeights = Backpropagation.GetGradientsForWeights(
                _InputSignals[(int)layerEventArgs.NumberOfActiveDataset], 
                _DeltasErrorsOfNeurons);

            _UpdatesForWeights = Backpropagation.GetUpdatesForWeights(
                _GradientsForWeights,
                _UpdatesForWeights,
                layerEventArgs.LearningRate,
                layerEventArgs.MomentumRate,
                layerEventArgs.LearningOptimizing);

            for (int synapse = 0; synapse < _SynapsesCount; synapse++)
            {
                for (int neuron = 0; neuron < _NeuronsCount; neuron++)
                {
                    _NeuronWeights[synapse][neuron] += _UpdatesForWeights[synapse][neuron];
                }
            }
        }

        protected virtual void UpdateWeightsForBias(LayerEventArgs layerEventArgs)
        {
            _UpdatesForBias = Backpropagation.GetUpdatesForBias(
                _DeltasErrorsOfNeurons,
                _UpdatesForBias,
                layerEventArgs.LearningRate,
                layerEventArgs.MomentumRate,
                layerEventArgs.LearningOptimizing);

            for (int i = 0; i < _BiasWeights.Length; i++)
            {
                _BiasWeights[i] += _UpdatesForBias[i];
            }
        }

        protected abstract void CalculateDeltasErrorsForThisLayer(LayerEventArgs nextLayerEventArgs);
    }
}
