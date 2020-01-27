using System.Collections.Generic;

namespace NeuralNetwork.LayersFactory
{
    interface ILayersFactory
    {
        LinkedList<ILayer> CreateLayers();
    }
}
