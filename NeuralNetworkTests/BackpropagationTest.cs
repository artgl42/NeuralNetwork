using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork;
using NeuralNetwork.Core;

namespace NeuralNetworkTests
{
    [TestClass]
    public class BackpropagationTest
    {
        [TestMethod]
        public void GetErrorsOutputSignal()
        {
            //Arrange
            float[][] outputSignal = new float[][]
                {
                    new float[] { 1 },
                    new float[] { 0 },
                    new float[] { 0 },
                    new float[] { 1 }
                };

            float[][] actualSignal = new float[][]
            {
                    new float[] { 1 },
                    new float[] { 0 },
                    new float[] { 0 },
                    new float[] { 1 }
            };

            float[][] expected = new float[][]
            {
                new float[] { 0 },
                new float[] { 0 },
                new float[] { 0 },
                new float[] { 0 }
            };

            //Act
            var actual = NeuralMath.GetTotalLoss(outputSignal, actualSignal, LossFunc.MSE);

            //Assert
            for (int i = 0; i < expected.Length; i++)
            {
                for (int j = 0; j < expected[i].Length; j++)
                {
                    Assert.AreEqual(expected[i][j], actual);
                }
            }
        }

        [TestMethod]
        public void GetDeltasErrorsOutputLayer()
        {
            //Arrange
            float[] outputSignal = new float[]
                {
                    0.95F,
                    0.55F,
                    0.40F,
                    0.90F
                };

            float[] actualSignal = new float[]
            {
                    1,
                    0,
                    0,
                    1
            };

            float[] expected = new float[]
            {
                0.002375001F,
                -0.136125F,
                -0.096F,
                0.009000004F
            };

            //Act
            var actual = Backpropagation.GetDeltasOutputLayer(outputSignal, actualSignal, ActivationFunc.Sigmoid);

            //Assert
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i]);
            }
        }

        [TestMethod]
        public void GetDeltasErrorsHiddenLayer()
        {
            //Arrange
            float[] outputSignal = new float[]
                {
                    0.95F,
                    0.55F,
                    0.90F
                };

            float[][] weightsNextLayer = new float[][]
            {
                    new float[] { 1 },
                    new float[] { 1 },
                    new float[] { 1 },
            };

            float[] deltasErrorsNextLayer = new float[]
            {
                    0.15F
            };

            float[] expected = new float[]
            {
                0.007125002F,
                0.0371250026F,
                0.0135000031F
            };

            //Act
            var actual = Backpropagation.GetDeltasHiddenLayer(outputSignal, weightsNextLayer, deltasErrorsNextLayer, ActivationFunc.Sigmoid);

            //Assert
            for (int i = 0; i < expected.Length; i++)
            {
                    Assert.AreEqual(expected[i], actual[i]);
            }
        }

        [TestMethod]
        public void GetGradientsForWeights()
        {
            //Arrange
            float[] inputSignal = new float[]
                {
                    0.95F, 0.55F, 0.90F
                };

            float[] deltasErrorThisLayer = new float[]
            {
                    0.77F
            };

            float[][] expected = new float[][]
            {
                new float[] { 0.73149997F },
                new float[] { 0.4235F },
                new float[] { 0.692999959F }
            };

            //Act
            var actual = Backpropagation.GetGradientsForWeights(inputSignal, deltasErrorThisLayer);

            //Assert
            for (int i = 0; i < expected.Length; i++)
            {
                for (int j = 0; j < expected[i].Length; j++)
                {
                    Assert.AreEqual(expected[i][j], actual[i][j]);
                }
            }
        }

        [TestMethod]
        public void GetUpdatesForWeights()
        {
            //Arrange
            float[][] gradientsForWeights = new float[][]
            {
                new float[] { 0.73F },
                new float[] { 0.42F },
                new float[] { 0.70F }
            };

            float[][] updatesForWeightsOld = new float[][]
            {
                new float[] { 0.14F },
                new float[] { 0.17F },
                new float[] { 0.19F }
            };

            float[][] expected = new float[][]
            {
                new float[] { 0.87F },
                new float[] { 0.59F },
                new float[] { 0.89F }
            };

            //Act
            var actual = Backpropagation.GetUpdatesForWeights(gradientsForWeights, updatesForWeightsOld, 1, 1, LearningOptimizing.NAG);

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
