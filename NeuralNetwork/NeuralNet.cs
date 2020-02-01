using NeuralNetwork.Core;
using NeuralNetwork.LayersFactory;
using NeuralNetwork.NeuralEventArgs;
using System;
using System.Collections.Generic;
using NLog;
using System.IO;

namespace NeuralNetwork
{
    // TODO: SAVE and LOAD
    // TODO: индесатор для NeuralLayers (возможность удалять слой) ???
    // TODO: обработку исключений, кидать все ошибки в текстовый файл и завершать программу
    // TODO: LearningCounter -> MaxValue свойство, чтобы предотвратить OverflowException при преобразовании в Int32 значение
    // TODO: проверку параметров в классе Settings -> CreateNeuralNet
    // TODO: методы обработки различных форматов данных для загружаемого датасета

    public enum InitializerWeights { Random, Zeros, Ones, XavierUniform }
    public enum InitializerBias { Zeros, Ones, Random }
    public enum ActivationFunc { Sigmoid, TanH, Identity, ReLU, Gaussian }
    public enum LossFunc { MSE, RootMSE, Arctan }
    public enum LearningOptimizing { SGDM }

    [Serializable]
    public class NeuralNet : INeuralNet
    {
        //private static readonly string _saveDirPath = $@"{Directory.GetCurrentDirectory()}\save";
        //private static readonly string _saveFileName = "SavedNeuralNet_";
        //private static readonly string _dateTimeFormat = "dd-MM-yyyy_HH-mm-ss";
        //private static readonly string _saveFileExt = ".dat";
        private static readonly float _defaultLearningRate = 0.1F;
        private static readonly float _defaultMomentumRate = 0.9F;
        //private static readonly Logger _Logger = LogManager.GetCurrentClassLogger();
        private LinkedList<ILayer> _NeuralLayers { get; set; }

        // Нужна ли переменная _LastSavedNeuralNetName?
        //private string _LastSavedNeuralNetName { get; set; }
        // -----

        public ActivationFunc ActivationFunc { get; set; }
        public LearningOptimizing? LearningOptimizing { get; set; }
        public float LearningRate { get; set; }
        public float MomentumRate { get; set; }
        public LossFunc LossFunc { get; set; }
        public uint LearningCounter { get; private set; }

        #region ----- Constructor (Fluent Builder) ----- 
        private NeuralNet(Builder settings)
        {
            ActivationFunc = settings.ActivationFunc;
            LearningOptimizing = settings.LearningOptimizing ?? default(LearningOptimizing);
            LearningRate = settings.LearningRate ?? _defaultLearningRate;
            MomentumRate = settings.MomentumRate ?? _defaultMomentumRate;
            LossFunc = settings.LossFunc;
            var layersFactory = new NeuralLayerFactory(
                settings.InputNeurons, 
                settings.NeuronLayers.ToArray(), 
                settings.IsBiasNeurons,
                settings.WeightsInitializer,
                settings.BiasInitializer);
            _NeuralLayers = layersFactory.CreateLayers();
        }

        public class Builder
        {
            public uint InputNeurons { get; set; }
            public List<uint> NeuronLayers { get; set; }
            public bool IsBiasNeurons { get; set; }
            public float? LearningRate { get; private set; }
            public float? MomentumRate { get; private set; }
            public InitializerWeights WeightsInitializer { get; set; }
            public InitializerBias BiasInitializer { get; set; }
            public ActivationFunc ActivationFunc { get; private set; }
            public LossFunc LossFunc { get; private set; }
            public LearningOptimizing? LearningOptimizing { get; private set; } = null;

            public Builder()
            {
                NeuronLayers = new List<uint>();
            }

            public Builder SetNeuronsInputLayer(uint inputNeurons)
            {
                InputNeurons = inputNeurons;
                return this;
            }

            public Builder SetNeuronsForLayers(params uint[] neuronLayers)
            {
                NeuronLayers.AddRange(neuronLayers);
                return this;
            }

            public Builder SetBiasNeurons(bool isBiasNeurons, InitializerBias biasInitializer = default(InitializerBias))
            {
                IsBiasNeurons = isBiasNeurons;
                BiasInitializer = biasInitializer;
                return this;
            }

            public Builder SetLearningRate(float learningRate)
            {
                this.LearningRate = learningRate;
                return this;
            }

            public Builder SetMomentumRate(float momentumRate)
            {
                this.MomentumRate = momentumRate;
                return this;
            }

            public Builder SetWeightsInitializer(InitializerWeights weightsInitializer)
            {
                WeightsInitializer = weightsInitializer;
                return this;
            }

            public Builder SetActivationFunc(ActivationFunc activationFunc)
            {
                ActivationFunc = activationFunc;
                return this;
            }

            public Builder SetLossFunc(LossFunc lossFunc)
            {
                LossFunc = lossFunc;
                return this;
            }

            public Builder SetLearningOptimizing(LearningOptimizing learningOptimizing)
            {
                LearningOptimizing = learningOptimizing;
                return this;
            }

            public NeuralNet Build()
            {
                return new NeuralNet(this);
            }
        }
        #endregion

        public float[][] Activate(float[][] inputSignals)
        {            
            return OnActivationNeuralNet(inputSignals).OutputsSignalsPrevLayer;
        }

        public float Learn(float[][] inputSignals, float[][] expectedSignals, uint epochsCount)
        {
            OnActivationNeuralNet(inputSignals);
            for (int epoch = 1; epoch <= epochsCount; epoch++)
            {
                LearningProcess(inputSignals, expectedSignals);
            }
            return CalculateError(inputSignals, expectedSignals);
        }

        public void Learn(float[][] inputSignals, float[][] expectedSignals, double acceptLoss)
        {
            float _currentLoss;
            OnActivationNeuralNet(inputSignals);
            do
            {
                LearningProcess(inputSignals, expectedSignals);
                _currentLoss = CalculateError(inputSignals, expectedSignals);
            } while (_currentLoss > acceptLoss);
        }

        public IEnumerable<float> Learn(float[][] inputSignals, float[][] expectedSignals, uint epochsCount, uint returnLossPeriod = 100)
        {
            OnActivationNeuralNet(inputSignals);
            for (int epoch = 1; epoch <= epochsCount; epoch++)
            {
                LearningProcess(inputSignals, expectedSignals);
                if (LearningCounter % returnLossPeriod == 0) yield return CalculateError(inputSignals, expectedSignals);
            }
        }

        public IEnumerable<float> Learn(float[][] inputSignals, float[][] expectedSignals, double acceptLoss = 0.01, uint returnLossPeriod = 100)
        {
            float _currentLoss;
            OnActivationNeuralNet(inputSignals);
            do
            {
                LearningProcess(inputSignals, expectedSignals);
                _currentLoss = CalculateError(inputSignals, expectedSignals);
                if (LearningCounter % returnLossPeriod == 0) yield return _currentLoss;           
            } while (_currentLoss > acceptLoss);
        }

        public float CalculateError(float[][] inputSignals, float[][] expectedSignals)
        {
            var _outputSignals = OnActivationNeuralNet(inputSignals).OutputsSignalsPrevLayer;
            return NeuralMath.GetTotalLoss(_outputSignals, expectedSignals, LossFunc);
        }

        //public string SaveNeuralNetToFile()
        //{
        //    if (!Directory.Exists(_saveDirPath)) Directory.CreateDirectory(_saveDirPath);
        //    string dateTimeNow = DateTime.Now.ToString(_dateTimeFormat);
        //    string filePath = $@"{_saveDirPath}\{_saveFileName}{dateTimeNow}{_saveFileExt}";
        //    using (FileStream stream = new FileStream(filePath, FileMode.CreateNew))
        //    {
        //        new BinaryFormatter().Serialize(stream, _NeuralNet);
        //        _LastSavedNeuralNetName = filePath;
        //        return filePath;
        //    }
        //}

        //public bool LoadNeuralNetFromFile(string filePath = null)
        //{
        //    filePath = filePath ?? _LastSavedNeuralNetName;
        //    if (File.Exists(filePath))
        //    {
        //        using (FileStream stream = File.Open(filePath, FileMode.Open))
        //        {
        //            _NeuralNet = (NeuralNet)new BinaryFormatter().Deserialize(stream);
        //            return true;
        //        }
        //    }
        //    else
        //    {
        //        return false;
        //    }
        //}

        private void LearningProcess(float[][] inputSignals, float[][] expectedSignals)
        {
            for (int numberOfActiveDataset = 0; numberOfActiveDataset < inputSignals.Length; numberOfActiveDataset++)
            {
                OnActivationNeuralNet(inputSignals, numberOfActiveDataset);
                OnLearningNeuralNet(expectedSignals, numberOfActiveDataset);
            }
            LearningCounter++;
        }

        private LayerEventArgs OnActivationNeuralNet(float[][] inputsSignals, int? numberOfActiveDataset = null)
        {
            try
            {
                LayerEventArgs activationEventArgs = new LayerEventArgs();
                activationEventArgs.ActivationFunc = ActivationFunc;
                activationEventArgs.OutputsSignalsPrevLayer = inputsSignals;
                activationEventArgs.NumberOfActiveDataset = numberOfActiveDataset;
                (_NeuralLayers.First.Value as NeuralLayer).Activate(this, activationEventArgs);
                return activationEventArgs;
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.DarkRed;
                Console.WriteLine("\nInnerException: " + ex.InnerException);
                Console.WriteLine("\nMessage: " + ex.Message);
                Console.WriteLine("\nSource: " + ex.Source);
                Console.WriteLine("\nTargetSite: " + ex.TargetSite);
                Console.WriteLine("\nStackTrace: ");
                Console.WriteLine("\n" + ex.StackTrace);
                Console.ReadKey();

                //Logger.Fatal(exception.Message);
                //LogManager.Flush();
                //LogManager.Shutdown();
                //Environment.Exit(0);
            }
            return null;
        }

        private void OnLearningNeuralNet(float[][] expectedSignals, int numberOfActiveDataset)
        {
            try
            {
                LayerEventArgs learningEventArgs = new LayerEventArgs();
                learningEventArgs.ActivationFunc = ActivationFunc;
                learningEventArgs.LearningOptimizing = LearningOptimizing;
                learningEventArgs.LearningRate = LearningRate;
                learningEventArgs.MomentumRate = MomentumRate;
                learningEventArgs.ExpectedSignalsOutLayer = expectedSignals;
                learningEventArgs.NumberOfActiveDataset = numberOfActiveDataset;
                (_NeuralLayers.Last.Value as NeuralLayer).Learning(this, learningEventArgs);
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.DarkRed;
                Console.WriteLine("\nInnerException: " + ex.InnerException);
                Console.WriteLine("\nMessage: " + ex.Message);
                Console.WriteLine("\nSource: " + ex.Source);
                Console.WriteLine("\nTargetSite: " + ex.TargetSite);
                Console.WriteLine("\nStackTrace: ");
                Console.WriteLine("\n" + ex.StackTrace);
                Console.ReadKey();

                //Logger.Fatal(exception.Message);
                //LogManager.Flush();
                //LogManager.Shutdown();
                //Environment.Exit(0);
            }
        }
    }
}
