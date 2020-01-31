using System.Collections.Generic;

namespace NeuralNetwork
{
    public interface INeuralNet
    {
        ActivationFunc ActivationFunc { get; set; }
        LearningOptimizing? LearningOptimizing { get; set; }
        float LearningRate { get; set; }
        float MomentumRate { get; set; }
        LossFunc LossFunc { get; set; }
        uint LearningCounter { get; }

        float[][] Activate(float[][] inputSignals);
        float Learn(float[][] inputSignals, float[][] expectedSignals, uint epochsCount);
        void Learn(float[][] inputSignals, float[][] expectedSignals, double acceptLoss);
        IEnumerable<float> Learn(float[][] inputSignals, float[][] expectedSignals, uint epochsCount, uint returnLossPeriod = 100);
        IEnumerable<float> Learn(float[][] inputSignals, float[][] expectedSignals, double loss = 0.01, uint returnLossPeriod = 100);
        float CalculateError(float[][] inputSignals, float[][] expectedSignals);
    }
}
