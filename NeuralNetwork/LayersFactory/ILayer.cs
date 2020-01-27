using NeuralNetwork.NeuralEventArgs;
using System;

namespace NeuralNetwork.LayersFactory
{
    interface ILayer
    {
        event EventHandler<LayerEventArgs> ActivatingLayer;
        event EventHandler<LayerEventArgs> LearningLayer;
        void Activate(object sender, LayerEventArgs prevLayerEventArgs);
        void Learning(object sender, LayerEventArgs nextLayerEventArgs);
    }
}
