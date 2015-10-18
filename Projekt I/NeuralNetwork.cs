using Encog;
using Encog.Engine.Network.Activation;
using Encog.ML;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Data.Versatile;
using Encog.ML.Data.Versatile.Columns;
using Encog.ML.Data.Versatile.Sources;
using Encog.ML.Factory;
using Encog.ML.Model;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Util.CSV;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace sieci_neuronowe
{
    public class NeuralNetwork
    {
        private const string FirstColumn = "x";
        private const string SecondColumn = "y";
        private const string PredictColumn = "cls";

        private readonly string _learningPath;
        private readonly string _testingPath;
        private readonly string _logOutputPath;
        private readonly Random _rng;
        private const int RandomnessSeed = 1001;
        private const string GeneratedImagePath = @".\area_classification.bmp";

        public NeuralNetwork(CommandLineParser parser)
            : this(parser.LearningSetFilePath, parser.TestingSetFilePath, parser.LogFilePath)
        { }

        public NeuralNetwork(string learningPath, string testingPath, string logOutputPath)
        {
            _testingPath = testingPath ?? learningPath;
            _learningPath = learningPath;
            _logOutputPath = logOutputPath;
            _rng = new Random(RandomnessSeed);
        }

        public void Run()
        {
            // Loop over the entire, original, dataset and feed it through the model.
            // This also shows how you would process new data, that was not part of your
            // training set.  You do not need to retrain, simply use the NormalizationHelper
            // class.  After you train, you can save the NormalizationHelper to later
            // normalize and denormalize your data.

            var csvLearningDataSource = new CSVDataSource(_learningPath, true, CSVFormat.DecimalPoint);
            var dataSet = PrepareDataSet(csvLearningDataSource);
            csvLearningDataSource.Close();

            // Create feedforward neural network as the model type. MLMethodFactory.TYPE_FEEDFORWARD.
            // You could also other model types, such as:
            // MLMethodFactory.SVM:  Support Vector Machine (SVM)
            // MLMethodFactory.TYPE_RBFNETWORK: RBF Neural Network
            // MLMethodFactor.TYPE_NEAT: NEAT Neural Network
            // MLMethodFactor.TYPE_PNN: Probabilistic Neural Network

            var trainingModel = new EncogModel(dataSet);
            trainingModel.SelectMethod(dataSet, MLMethodFactory.TypeFeedforward);

            // Send any output to the console.

            using (var writetext = new StreamWriter(_logOutputPath))
            {
                trainingModel.Report = new StreamStatusReportable(writetext);

                // Now normalize the data.  Encog will automatically determine the correct normalization
                // type based on the model you chose in the last step.

                dataSet.Normalize();

                // Hold back some data for a final validation.
                // Shuffle the data into a random ordering.
                // Use a seed of 1001 so that we always use the same holdback and will get more consistent results.

                trainingModel.HoldBackValidation(0.1, true, RandomnessSeed);

                // Choose whatever is the default training type for this model.

                trainingModel.SelectTrainingType(dataSet);

                var network = CreateNetwork(_rng);

                // TODO: Ma być online, tzn. training dataset pusty (niemożliwe z tą implementacją?)
                var backpropagation = new Backpropagation(network, dataSet, 0.00001, 0.01);
                const int iterationCount = 10000;
                for (var i = 0; i < iterationCount; i++)
                {
                    backpropagation.Iteration();
                    if (i % (iterationCount / 10) != 0) continue;
                    var err = backpropagation.Error;
                    writetext.WriteLine("Backpropagation error: " + err);
                    Console.WriteLine("Iteration progress: " + i + "/" + iterationCount + ", error: " + err);
                }

                PrintWeights(network, writetext);
                var usedMethod = (IMLRegression)backpropagation.Method;

                // Use a 5-fold cross-validated train.  Return the best method found.
                // var usedMethod = (IMLRegression)trainingModel.Crossvalidate(5, true);

                // Display the training and validation errors.

                writetext.WriteLine("Training error: " + trainingModel.CalculateError(usedMethod, trainingModel.TrainingDataset));
                writetext.WriteLine("Validation error: " + trainingModel.CalculateError(usedMethod, trainingModel.ValidationDataset));

                // Display our normalization parameters.

                var normHelper = dataSet.NormHelper;
                writetext.WriteLine(normHelper);

                // Display the final model.

                writetext.WriteLine("Final model: " + usedMethod);

                var allPoints = new List<NeuroPoint>();

                TestData(_learningPath, normHelper, usedMethod, allPoints);
                TestData(_testingPath, normHelper, usedMethod, allPoints);
                PrintPoints(allPoints, writetext);

                writetext.WriteLine("Training error: " + trainingModel.CalculateError(usedMethod, trainingModel.TrainingDataset));
                writetext.WriteLine("Validation error: " + trainingModel.CalculateError(usedMethod, trainingModel.ValidationDataset));

                PictureGenerator.DrawArea(GeneratedImagePath, usedMethod, allPoints, normHelper, 1024, 1024);
            }

            EncogFramework.Instance.Shutdown();
        }

        private static BasicNetwork CreateNetwork(Random rng)
        {
            var network = new BasicNetwork();

            // TODO: Wszystkie parametry konfigurowalne dla każdego layera (poza pierwszym bo input?)
            network.AddLayer(new BasicLayer(new ActivationLinear(), true, 2));
            network.AddLayer(new BasicLayer(new ActivationLinear(), true, 3));
            network.AddLayer(new BasicLayer(new ActivationLinear(), true, 1));
            network.Structure.FinalizeStructure();

            // Zrób pełną sieć
            for (var i = 0; i < network.LayerCount - 1; i++)
            {
                for (var j = 0; j < network.GetLayerNeuronCount(i); j++)
                {
                    for (var k = 0; k < network.GetLayerNeuronCount(i + 1); k++)
                    {
                        network.SetWeight(i, j, k, (rng.NextDouble() - 0.5) * 0.1);
                    }
                }
            }

            return network;
        }

        private static VersatileMLDataSet PrepareDataSet(IVersatileDataSource dataSource)
        {
            var dataSet = new VersatileMLDataSet(dataSource);
            dataSet.DefineSourceColumn(FirstColumn, 0, ColumnType.Continuous);
            dataSet.DefineSourceColumn(SecondColumn, 1, ColumnType.Continuous);

            // Column that we are trying to predict.
            var outputColumnDefinition = dataSet.DefineSourceColumn(PredictColumn, 2, ColumnType.Nominal);

            // Analyze the data, determine the min/max/mean/sd of every column.
            dataSet.Analyze();

            // Map the prediction column to the output of the model, and all other columns to the input.
            dataSet.DefineSingleOutputOthersInput(outputColumnDefinition);
            dataSet.LagWindowSize = 1;
            return dataSet;
        }

        private static void PrintPoints(IEnumerable<NeuroPoint> points, StreamWriter writetext)
        {
            foreach (var point in points)
            {
                var result = new StringBuilder();

                result.AppendFormat(
                    "({0: 0.00;-0.00}, {1: 0.00;-0.00}) -> predicted: {2}",
                    point.X,
                    point.Y,
                    point.Category);
                if (point.Correct >= 0)
                {
                    result.AppendFormat("(correct: {0})", point.Correct);
                }

                writetext.WriteLine(result);
            }
        }

        private static void PrintWeights(BasicNetwork network, TextWriter writer)
        {
            for (int i = 0; i < network.LayerCount - 1; i++)
            {
                writer.WriteLine("Layer {0}, neurons number {1}", i, network.GetLayerNeuronCount(i));
                for (var j = 0; j < network.GetLayerNeuronCount(i); j++)
                {
                    for (var k = 0; k < network.GetLayerNeuronCount(i + 1); k++)
                    {
                        writer.WriteLine("{0}->{1}: {2:F3}", j, k, network.GetWeight(i, j, k));
                    }
                }
            }
        }

        private static void TestData(
            string testedPath,
            NormalizationHelper helper,
            IMLRegression usedMethod,
            ICollection<NeuroPoint> results)
        {
            var csv = new ReadCSV(testedPath, true, CSVFormat.DecimalPoint);

            while (csv.Next())
            {
                var x = csv.GetDouble(0);
                var y = csv.GetDouble(1);
                var correct = -1;
                if (csv.ColumnCount > 2)
                {
                    correct = (int)csv.GetDouble(2);
                }

                var data = new BasicMLData(new[] { x, y });
                helper.NormalizeInputVector(new[] { csv.Get(0), csv.Get(1) }, data.Data, false);
                IMLData output = new BasicMLData(new[] { x, y, usedMethod.Compute(data)[0] });
                string stringChosen = helper.DenormalizeOutputVectorToString(output)[0];
                var computed = int.Parse(stringChosen);
                results.Add(new NeuroPoint(x, y, computed, correct));
            }

            csv.Close();
        }
    }
}