using System;

namespace NeuralNetwork.NeuralEventArgs
{
    internal class LayerEventArgs: EventArgs
    {
        internal ActivationFunc ActivationFunc { get; set; }
        internal LearningOptimizing? LearningOptimizing { get; set; }
        internal float LearningRate { get; set; }
        internal float MomentumRate { get; set; }
        internal float[][] OutputsSignalsPrevLayer { get; set; }
        internal float[][] ExpectedSignalsOutLayer { get; set; }
        internal float[][] NeuronsWeightsNextLayer { get; set; }
        internal float[] DeltasErrorsNextLayer { get; set; }
        internal int? NumberOfActiveDataset { get; set; }
    }
}
