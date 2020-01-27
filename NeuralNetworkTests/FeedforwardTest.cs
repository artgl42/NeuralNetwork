using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork;
using NeuralNetwork.Core;

namespace NeuralNetworkTests
{
    [TestClass]
    public class FeedforwardTest
    {
        [TestMethod]
        public void GetOutputsSignals()
        {
            //Arrange
            float[][] inputSignal = new float[][]
                {
                    new float[] { 0, 0 },
                    new float[] { 0, 1 },
                    new float[] { 1, 0 },
                    new float[] { 1, 1 }
                };

            float[][] neuronsWeights = new float[][]
            {
                    new float[] { 1, 1, 1 },
                    new float[] { 1, 1, 1 }
            };

            float[][] expected = new float[][]
            {
                new float[] { 0.5F, 0.5F, 0.5F },
                new float[] { 0.7310586F, 0.7310586F, 0.7310586F },
                new float[] { 0.7310586F, 0.7310586F, 0.7310586F },
                new float[] { 0.8807971F, 0.8807971F, 0.8807971F }
            };


            //Act
            var actual = Feedforward.GetOutputSignalsOfAllSamples(inputSignal, neuronsWeights, ActivationFunc.Sigmoid, null);

            //Assert
            for (int i = 0; i < expected.Length; i++)
            {
                for (int j = 0; j < expected[i].Length; j++)
                {
                    Assert.AreEqual(expected[i][j], actual[i][j]);
                }
            }          
        }
    }
}
